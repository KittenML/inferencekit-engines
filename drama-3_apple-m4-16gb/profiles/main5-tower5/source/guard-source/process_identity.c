/* SPDX-License-Identifier: Apache-2.0 */
#include <libproc.h>
#include <stdint.h>
#include <sys/proc_info.h>
#include <sys/proc.h>

int ik_process_identity(int pid, uint64_t values[4]) {
    struct proc_bsdinfo info;
    if (pid <= 1 || proc_pidinfo(pid, PROC_PIDTBSDINFO, 0, &info, sizeof(info)) != (int)sizeof(info)) return 0;
    if (info.pbi_status == SZOMB) return 0;
    values[0] = info.pbi_pid;
    values[1] = info.pbi_uid;
    values[2] = info.pbi_start_tvsec;
    values[3] = info.pbi_start_tvusec;
    return info.pbi_start_tvsec != 0;
}
