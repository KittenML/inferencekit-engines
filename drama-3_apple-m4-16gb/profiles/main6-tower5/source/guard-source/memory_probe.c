/* SPDX-License-Identifier: Apache-2.0 */
#include <libproc.h>
#include <sys/resource.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <errno.h>
#include <limits.h>
int main(int argc, char **argv) {
    if (argc != 2) return 2;
    char *end = NULL;
    long requested = strtol(argv[1], &end, 10);
    if (!end || *end || requested <= 0 || requested > INT_MAX) return 2;
    struct rusage_info_v4 value = {0};
    int result = proc_pid_rusage((int)requested, RUSAGE_INFO_V4, (rusage_info_t *)&value);
    if (result) { printf("{\"ok\":false,\"errno\":%d}\n", errno); return 1; }
    printf("{\"ok\":true,\"source\":\"proc_pid_rusage RUSAGE_INFO_V4\","
           "\"resident_bytes\":%llu,\"wired_bytes\":%llu,\"physical_footprint_bytes\":%llu,"
           "\"lifetime_max_physical_footprint_bytes\":%llu,\"interval_max_physical_footprint_bytes\":%llu,"
           "\"pageins\":%llu,\"disk_read_bytes\":%llu,\"disk_written_bytes\":%llu}\n",
           (unsigned long long)value.ri_resident_size, (unsigned long long)value.ri_wired_size,
           (unsigned long long)value.ri_phys_footprint, (unsigned long long)value.ri_lifetime_max_phys_footprint,
           (unsigned long long)value.ri_interval_max_phys_footprint, (unsigned long long)value.ri_pageins,
           (unsigned long long)value.ri_diskio_bytesread, (unsigned long long)value.ri_diskio_byteswritten);
    return 0;
}
