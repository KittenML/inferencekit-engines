#pragma once

#include <cerrno>
#include <cstdint>
#include <cstdio>
#include <csignal>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>
#include <fcntl.h>
#include <sys/prctl.h>
#include <sys/wait.h>
#include <unistd.h>

class KokoroPcmProcess {
public:
    KokoroPcmProcess(const std::string &directory, const std::vector<std::string> &command) {
        if (command.empty()) {
            throw std::invalid_argument("the Kokoro PCM worker needs an executable");
        }
        std::vector<char *> arguments;
        for (const std::string &argument : command) {
            arguments.push_back(const_cast<char *>(argument.c_str()));
        }
        arguments.push_back(nullptr);
        Pipe input, audio, messages;
        child_ = fork();
        if (child_ < 0) {
            throw std::runtime_error("could not spawn the Kokoro PCM worker");
        }
        if (child_ == 0) {
            prctl(PR_SET_PDEATHSIG, SIGTERM);
            if (dup2(input.read_end, STDIN_FILENO) < 0 || dup2(audio.write_end, STDOUT_FILENO) < 0 ||
                dup2(messages.write_end, STDERR_FILENO) < 0 || chdir(directory.c_str()) != 0) {
                _exit(126);
            }
            execv(arguments.front(), arguments.data());
            _exit(127);
        }
        try {
            input_ = open(input.write_end, "w");
            audio_ = open(audio.read_end, "r");
            messages_ = open(messages.read_end, "r");
            input.close_unused();
            audio.close_unused();
            messages.close_unused();
            std::string message;
            while (next_message(message)) {
                if (message.rfind("ready.", 0) == 0) {
                    return;
                }
            }
            throw std::runtime_error("the Kokoro PCM worker did not reach runtime readiness");
        } catch (...) {
            stop();
            throw;
        }
    }

    ~KokoroPcmProcess() { stop(); }

    std::vector<int16_t> synthesize(const std::string &text) {
        if (text.empty() || text.find_first_of("\r\n") != std::string::npos ||
            std::fprintf(input_.get(), "%s\n", text.c_str()) < 0 || std::fflush(input_.get()) != 0) {
            throw std::runtime_error("could not send the sentence to the Kokoro PCM worker");
        }
        std::vector<int16_t> pcm;
        std::string message;
        while (next_message(message)) {
            if (message.rfind("KOKORO_UTT_END", 0) == 0) {
                if (pcm.empty()) {
                    throw std::runtime_error("the Kokoro PCM worker returned no PCM");
                }
                return pcm;
            }
            size_t frames = 0;
            int rate = 0;
            if (std::sscanf(message.c_str(), "KOKORO_PCM_BEGIN %zu %d", &frames, &rate) == 2) {
                if (rate != 24000 || frames == 0 || frames > 24000 * 300 - pcm.size()) {
                    throw std::runtime_error("invalid Kokoro PCM frame count or sample rate");
                }
                const size_t offset = pcm.size();
                pcm.resize(offset + frames);
                if (std::fread(pcm.data() + offset, sizeof(int16_t), frames, audio_.get()) != frames) {
                    throw std::runtime_error("truncated Kokoro PCM");
                }
            }
        }
        throw std::runtime_error("the Kokoro PCM worker exited during synthesis");
    }

private:
    struct Pipe {
        Pipe() {
            int ends[2];
            if (pipe2(ends, O_CLOEXEC) != 0) {
                throw std::runtime_error("could not create Kokoro PCM pipes");
            }
            read_end = ends[0];
            write_end = ends[1];
        }

        ~Pipe() { close_unused(); }
        void close_unused() {
            for (int *end : {&read_end, &write_end}) {
                if (*end >= 0) {
                    close(*end);
                    *end = -1;
                }
            }
        }
        int read_end = -1;
        int write_end = -1;
    };

    struct CloseFile {
        void operator()(FILE *file) const { std::fclose(file); }
    };

    using File = std::unique_ptr<FILE, CloseFile>;
    static File open(int &descriptor, const char *mode) {
        auto *file = fdopen(descriptor, mode);
        if (!file) {
            throw std::runtime_error("could not open a Kokoro PCM pipe");
        }
        descriptor = -1;
        return File(file);
    }

    bool next_message(std::string &message) {
        char line[2048];
        if (!std::fgets(line, sizeof(line), messages_.get())) {
            return false;
        }
        message = line;
        std::fputs(line, stderr);
        // The OpenCL logger can prefix a protocol record with a literal escaped newline.
        const size_t separator = message.rfind("\\n");
        if (separator != std::string::npos) {
            const size_t record = separator + 2;
            if (message.compare(record, 6, "ready.") == 0 ||
                message.compare(record, 17, "KOKORO_PCM_BEGIN ") == 0 ||
                message.compare(record, 14, "KOKORO_UTT_END") == 0) {
                message.erase(0, record);
            }
        }
        return true;
    }

    void stop() {
        input_.reset();
        audio_.reset();
        messages_.reset();
        if (child_ > 0) {
            kill(child_, SIGTERM);
            while (waitpid(child_, nullptr, 0) < 0 && errno == EINTR) {
            }
            child_ = -1;
        }
    }

    pid_t child_ = -1;
    File input_;
    File audio_;
    File messages_;
};
