/*
 * rpiz - https://github.com/bp0/rpiz
 * Copyright (C) 2017  Burt P. <pburt0@gmail.com>
 *
 * This program is free software; you can redistribute it and/or
 * modify it under the terms of the GNU General Public License
 * as published by the Free Software Foundation; either version 2
 * of the License, or (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program; if not, write to the Free Software
 * Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301, USA.
 *
 */

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include "util.h"
#include "cpu_riscv.h"

#define MAX_CORES 128

static int search_for_flag(char *flags, const char *flag) {
    char *p = strstr(flags, flag);
    int l = strlen(flag);
    int front = 0, back = 0;
    //DEBUG printf("search_for_flag( %x, \"%s\")\n", flags, flag);
    if (strlen(flag) == 0 || strchr(flag, ' ') )
        return 0;
    while (p) {
        if (p == flags) front = 1;
        else if (*(p - 1) == ' ') front = 1;
        if (*(p + l) == ' ' || *(p + l) == '\0')
            back = 1;
        if (front && back)
            return 1;
        p = strstr(p + l, flag);
    }
    return 0;
}

typedef struct {
    int id; /* hart */
    int cpukhz_min, cpukhz_max, cpukhz_cur;

    /* point to a cpu_string.str */
    char *model_name;
    char *isa;
    char *flags; /* extensions as flag list */
    char *cpukhz_max_str;
} riscv_core;

struct riscv_proc {
    cpu_string_list *model_name;
    cpu_string_list *isa;
    cpu_string_list *flags;
    cpu_string_list *cpukhz_max_str;

    cpu_string_list *each_flag;

    char cpu_name[256];
    char *cpu_desc;
    int max_khz;
    int core_count;
    riscv_core cores[MAX_CORES];

    rpiz_fields *fields;
};

#define CHECK_FOR(k) (strncmp(k, key, (strlen(k) < strlen(key)) ? strlen(k) : strlen(key)) == 0)
#define GET_STR(k, s) if (CHECK_FOR(k)) { p->cores[core].s = strlist_add(p->s, value); continue; }
#define FIN_PROC() if (core >= 0) if (!p->cores[core].model_name) { p->cores[core].model_name = strlist_add(p->model_name, rep_pname); }

#define REDUP(f) if(p->cores[di].f && !p->cores[i].f) { p->cores[i].f = strlist_add(p->f, p->cores[di].f); }

#ifndef PROC_CPUINFO
#define PROC_CPUINFO "/proc/cpuinfo"
#endif

static int scan_cpu(riscv_proc* p) {
    kv_scan *kv; char *key, *value;
    int core = -1;
    int i, di;
    char rep_pname[256] = "RISC-V Processor";
    char tmp_maxfreq[128] = "";

    if (!p) return 0;

    kv = kv_new_file(PROC_CPUINFO);
    if (kv) {
        while( kv_next(kv, &key, &value) ) {
            if (CHECK_FOR("Processor")) {
                strcpy(rep_pname, value);
                continue;
            }

            if (CHECK_FOR("hart")) {
                FIN_PROC();
                core++;
                memset(&p->cores[core], 0, sizeof(riscv_core));
                p->cores[core].id = atoi(value);
                continue;
            }

            if (core < 0) {
                if ( CHECK_FOR("isa")
                     || CHECK_FOR("isa")
                     || CHECK_FOR("isa") ) {
                    /* this cpuinfo doesn't provide hart : n
                     * there is prolly only one core */
                    core++;
                    memset(&p->cores[core], 0, sizeof(riscv_core));
                    p->cores[core].id = 0;
                }
            }
            if (core >= 0) {
                GET_STR("model name", model_name);
                GET_STR("isa", isa);
            }
        }
        FIN_PROC();
        kv_free(kv);
    } else
        return 0;

    p->core_count = core + 1;

    /* re-duplicate missing data for /proc/cpuinfo variant that de-duplicated it */
    di = p->core_count - 1;
    for (i = di; i >= 0; i--) {
        if (p->cores[i].isa)
            di = i;
        else {
            REDUP(isa);
        }
    }

    /* data not from /proc/cpuinfo */
    for (i = 0; i < p->core_count; i++) {
        /* flags */
        p->cores[i].flags = riscv_isa_to_flags(p->cores[i].isa);

        /* freq */
        get_cpu_freq(p->cores[i].id, &p->cores[i].cpukhz_min, &p->cores[i].cpukhz_max, &p->cores[i].cpukhz_cur);
        sprintf(tmp_maxfreq, "%d", p->cores[i].cpukhz_max);
        p->cores[i].cpukhz_max_str = strlist_add(p->cpukhz_max_str, tmp_maxfreq);
        if (p->cores[i].cpukhz_max > p->max_khz)
            p->max_khz = p->cores[i].cpukhz_max;
    }

    return 1;
}

static char *gen_cpu_desc(riscv_proc *p) {
    char *ret = NULL;
    char tmp[1024];
    int i, l = 0;
    float maxfreq;
    if (p) {
        ret = malloc(4096);
        memset(ret, 0, 4096);
        for (i = 0; i < p->model_name->count; i++) {
            sprintf(tmp, "%dx %s", p->model_name->strs[i].ref_count, p->model_name->strs[i].str);
            sprintf(ret + l, "%s%s", (i>0) ? " + " : "", tmp);
            l += (i>0) ? strlen(tmp) + 3 : strlen(tmp);
        }
        sprintf(ret + l, "; "); l += 2;
        for (i = 0; i < p->cpukhz_max_str->count; i++) {
            maxfreq = atof(p->cpukhz_max_str->strs[i].str);
            if (maxfreq)
                maxfreq /= 1000;
            else
                maxfreq = 0.0f;
            sprintf(tmp, "%dx %0.2f MHz", p->cpukhz_max_str->strs[i].ref_count, maxfreq);
            sprintf(ret + l, "%s%s", (i>0) ? " + " : "", tmp);
            l += (i>0) ? strlen(tmp) + 3 : strlen(tmp);
        }
    }
    return ret;
}

#define APPEND_FLAG(f) strcat(all_flags, f); strcat(all_flags, " ");
static void process_flags(riscv_proc *s) {
    char flag[16] = "";
    char *all_flags; /* riscv_data.c: static char all_flags[1024] */
    char *cur, *next;
    int added_count = 0, i;
    if (!s) return;

    all_flags = (char*)riscv_ext_list();
    for(i = 0; i < s->flags->count; i++) {
        if (s->flags->strs[i].str) {
            cur = s->flags->strs[i].str;
            next = strchr(cur, ' '); if (!next) next = strchr(cur, '\0');
            while(next) {
                if (next-cur <= 15) {
                    memset(flag, 0, 16);
                    strncpy(flag, cur, next-cur);
                    if (strlen(flag) > 0) {
                        /* add it to the string list, copy the flag string's ref_count */
                        strlist_add_w(s->each_flag, flag, s->flags->strs[i].ref_count);

                        /* add it to the list of known all flags, if it isn't there */
                        if (!search_for_flag(all_flags, flag)) {
                            APPEND_FLAG(flag);
                            added_count++;
                        }
                    }
                }
                if (*next == '\0') break;
                cur = next + 1;
                next = strchr(cur, ' '); if (!next) next = strchr(cur, '\0');
            }
        }
    }
    // DEBUG printf("add_unknown_flags(): added %d previously unknown flags\n", added_count);
}

riscv_proc *riscv_proc_new(void) {
    riscv_proc *s = malloc( sizeof(riscv_proc) );
    if (s) {
        memset(s, 0, sizeof(*s));
        s->model_name = strlist_new();
        s->isa = strlist_new();
        s->flags = strlist_new();
        s->cpukhz_max_str = strlist_new();
        s->each_flag = strlist_new();
        if (!scan_cpu(s)) {
            riscv_proc_free(s);
            return NULL;
        }
        s->cpu_desc = gen_cpu_desc(s);
        process_flags(s);
    }
    return s;
}

void riscv_proc_free(riscv_proc *s) {
    if(s) {
        strlist_free(s->model_name);
        strlist_free(s->isa);
        strlist_free(s->flags);
        strlist_free(s->cpukhz_max_str);
        strlist_free(s->each_flag);
        fields_free(s->fields);
        free(s->cpu_desc);
        free(s);
    }
}

const char *riscv_proc_name(riscv_proc *s) {
    if (s)
        return s->cpu_name;
    else
        return NULL;
}

const char *riscv_proc_desc(riscv_proc *s) {
    if (s)
        return s->cpu_desc;
    else
        return NULL;
}

int riscv_proc_has_flag(riscv_proc *s, const char *flag) {
    int i;
    if (s && flag) {
        for (i = 0; i < s->each_flag->count; i++) {
            //DEBUG printf("(%s)...[%d/%d] %s %d\n", flag, i, s->each_flag->count, s->each_flag->strs[i].str, s->each_flag->strs[i].ref_count);
            if (strcmp(s->each_flag->strs[i].str, flag) == 0)
                return s->each_flag->strs[i].ref_count;
        }
    }
    return 0;
}

int riscv_proc_cores(riscv_proc *s) {
    if (s)
        return s->core_count;
    else
        return 0;
}

int riscv_proc_core_from_id(riscv_proc *s, int id) {
    int i = 0;
    if (s)
        for (i = 0; i < s->core_count; i++ )
            if (s->cores[i].id == id)
                return i;

    return -1;
}

int riscv_proc_core_id(riscv_proc *s, int core) {
    if (s)
        if (core >= 0 && core < s->core_count)
            return s->cores[core].id;

    return 0;
}

int riscv_proc_core_khz_min(riscv_proc *s, int core) {
    if (s)
        if (core >= 0 && core < s->core_count)
            return s->cores[core].cpukhz_min;

    return 0;
}

int riscv_proc_core_khz_max(riscv_proc *s, int core) {
    if (s)
        if (core >= 0 && core < s->core_count)
            return s->cores[core].cpukhz_max;

    return 0;
}

int riscv_proc_core_khz_cur(riscv_proc *s, int core) {
    if (s)
        if (core >= 0 && core < s->core_count) {
            get_cpu_freq(s->cores[core].id, NULL, NULL, &s->cores[core].cpukhz_cur);
            return s->cores[core].cpukhz_cur;
        }
    return 0;
}

static char* riscv_proc_cores_str(riscv_proc *s) {
    char *buff = NULL;
    if (s) {
        buff = malloc(128);
        if (buff)
            snprintf(buff, 127, "%d", riscv_proc_cores(s) );
    }
    return buff;
}

#define ADDFIELD(t, l, o, n, f) fields_update_bytag(s->fields, t, l, o, n, (rpiz_fields_get_func)f, (void*)s)
#define ADDFIELDSTR(t, l, o, n, str) fields_update_bytag(s->fields, t, l, o, n, NULL, (void*)str)
rpiz_fields *riscv_proc_fields(riscv_proc *s) {
    int i;
    char bn[256] = "", bt[256] = "", bv[256] = "", *bvp;
    if (s) {
        if (!s->fields) {
            /* first insert creates */
            s->fields =
            ADDFIELD("summary.proc_desc", 0, 0, "Proccesor", riscv_proc_desc );
            ADDFIELD("cpu.name",          0, 0, "Proccesor Name", riscv_proc_name );
            ADDFIELD("cpu.desc",          0, 0, "Proccesor Description", riscv_proc_desc );
            ADDFIELD("cpu.count",         0, 1, "Core Count", riscv_proc_cores_str );

            for(i = 0; i < s->core_count; i++) {
                sprintf(bt, "cpu.thread[%d].model_name", i);
                sprintf(bn, "[%d] linux name", s->cores[i].id);
                sprintf(bv, "%s", s->cores[i].model_name);
                bvp = strdup(bv); ADDFIELDSTR(bt, 0, 1, bn, bvp);

                sprintf(bt, "cpu.thread[%d].isa", i);
                sprintf(bn, "[%d] isa", s->cores[i].id);
                sprintf(bv, "%s", s->cores[i].isa);
                bvp = strdup(bv); ADDFIELDSTR(bt, 0, 1, bn, bvp);
            }

        }
        return s->fields;
    }
    return NULL;
}

