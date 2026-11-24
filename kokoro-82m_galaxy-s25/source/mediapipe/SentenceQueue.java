package org.inferencekit.mediapipecapture;

import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;
import java.util.PriorityQueue;

/** Work becomes eligible at its wave release and runs in order within each stream. */
final class SentenceQueue {
    static final class Request {
        final int index;
        long release;
        long deadline;
        long sequence;
        int part;

        Request(int index) {
            this.index = index;
        }
    }

    private final List<Request> requests = new ArrayList<>();
    private final PriorityQueue<Request> runnable;
    private final boolean[] occupied;
    private final int concurrency;
    private final long period;
    private int released;
    private int remaining;
    private long sequence;
    private boolean aborted;

    SentenceQueue(int concurrency, int count, long period, boolean deadlineOrder) {
        this.concurrency = concurrency;
        this.period = period;
        this.remaining = count;
        occupied = new boolean[concurrency];
        Comparator<Request> fifo = Comparator.comparingLong(request -> request.sequence);
        runnable = new PriorityQueue<>(deadlineOrder
                ? Comparator.comparingLong((Request request) -> request.deadline).thenComparing(fifo)
                : fifo);
        for (int index = 0; index < count; ++index) {
            requests.add(new Request(index));
        }
    }

    private void enqueue(Request request, long deadline) {
        request.deadline = deadline;
        request.sequence = sequence++;
        runnable.add(request);
    }

    synchronized void releaseWave(long release) {
        if (aborted) {
            return;
        }
        for (int slot = 0; slot < concurrency; ++slot) {
            Request request = requests.get(released++);
            request.release = release;
            if (!occupied[slot]) {
                occupied[slot] = true;
                enqueue(request, release + period);
            }
        }
        notifyAll();
    }

    synchronized Request take() throws InterruptedException {
        while (!aborted && remaining != 0 && runnable.isEmpty()) {
            wait();
        }
        return aborted ? null : runnable.poll();
    }

    synchronized void continueAt(Request request, long deadline) {
        ++request.part;
        if (!aborted) {
            enqueue(request, deadline);
            notifyAll();
        }
    }

    synchronized void finish(Request request) {
        --remaining;
        int next = request.index + concurrency;
        if (!aborted && next < released) {
            Request successor = requests.get(next);
            enqueue(successor, successor.release + period);
        } else {
            occupied[request.index % concurrency] = false;
        }
        notifyAll();
    }

    synchronized void abort() {
        aborted = true;
        notifyAll();
    }
}
