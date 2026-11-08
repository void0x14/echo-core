// main'den önce çalışır, temel x86-64 ile derlenir: derleme CPU'sundaki özellik
// bu işlemcide yoksa "illegal instruction" yerine anlaşılır hata verir.
// ECHO_GEREKLI_* tanımlarını build.zig hedef CPU'ya göre verir.
#include <cpuid.h>
#include <stdio.h>
#include <stdlib.h>

static int eksik = 0;
static void kontrol(int gerekli, unsigned reg, int bit, const char *ad) {
    if (!gerekli || (reg >> bit) & 1) return;
    if (!eksik) fputs("hata: bu program bu işlemcide çalışamaz; eksik CPU özelliği:", stderr);
    eksik = 1;
    fprintf(stderr, " %s", ad);
}

__attribute__((constructor)) static void echo_cpu_kontrol(void) {
    unsigned a1 = 0, b1 = 0, c1 = 0, d1 = 0, a7 = 0, b7 = 0, c7 = 0, d7 = 0;
    __get_cpuid_count(1, 0, &a1, &b1, &c1, &d1);
    __get_cpuid_count(7, 0, &a7, &b7, &c7, &d7);
    kontrol(ECHO_GEREKLI_avx, c1, 28, "avx");
    kontrol(ECHO_GEREKLI_fma, c1, 12, "fma");
    kontrol(ECHO_GEREKLI_f16c, c1, 29, "f16c");
    kontrol(ECHO_GEREKLI_avx2, b7, 5, "avx2");
    kontrol(ECHO_GEREKLI_bmi, b7, 3, "bmi");
    kontrol(ECHO_GEREKLI_bmi2, b7, 8, "bmi2");
    kontrol(ECHO_GEREKLI_avx512f, b7, 16, "avx512f");
    kontrol(ECHO_GEREKLI_avx512bw, b7, 30, "avx512bw");
    kontrol(ECHO_GEREKLI_avx512vl, b7, 31, "avx512vl");
    kontrol(ECHO_GEREKLI_avx512vnni, c7, 11, "avx512vnni");
    if (eksik) {
        fputs("\nBu makinede yeniden derleyin: zig build -Doptimize=ReleaseFast\n", stderr);
        exit(1);
    }
}
