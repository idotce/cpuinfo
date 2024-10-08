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
#include "board_dt.h"
#include "board_rpi.h"

static char unk[] = "(Unknown)";

/* information table from: http://elinux.org/RPi_HardwareHistory */
static struct {
    char *value, *intro, *model, *pcb, *mem, *mfg, *soc;
} rpi_boardinfo[] = {
/*  Value        Introduction  Model Name             PCB rev.  Memory(spec)   Manufacturer  SOC(spec) *
 *                             Raspberry Pi %s                                                         */
  { unk,         unk,          unk,                   unk,      unk,        unk,             NULL },
  { "Beta",      "Q1 2012",    "B (Beta)",            unk,      "256MB",    "(Beta board)",  NULL },
  { "0002",      "Q1 2012",    "B",                   "1.0",    "256MB",    unk,             "BCM2835" },
  { "0003",      "Q3 2012",    "B (ECN0001)",         "1.0",    "256MB",    "(Fuses mod and D14 removed)",   NULL },
  { "0004",      "Q3 2012",    "B",                   "2.0",    "256MB",    "Sony",          NULL },
  { "0005",      "Q4 2012",    "B",                   "2.0",    "256MB",    "Qisda",         NULL },
  { "0006",      "Q4 2012",    "B",                   "2.0",    "256MB",    "Egoman",        NULL },
  { "0007",      "Q1 2013",    "A",                   "2.0",    "256MB",    "Egoman",        NULL },
  { "0008",      "Q1 2013",    "A",                   "2.0",    "256MB",    "Sony",          NULL },
  { "0009",      "Q1 2013",    "A",                   "2.0",    "256MB",    "Qisda",         NULL },
  { "000d",      "Q4 2012",    "B",                   "2.0",    "512MB",    "Egoman",        NULL },
  { "000e",      "Q4 2012",    "B",                   "2.0",    "512MB",    "Sony",          NULL },
  { "000f",      "Q4 2012",    "B",                   "2.0",    "512MB",    "Qisda",         NULL },
  { "0010",      "Q3 2014",    "B+",                  "1.0",    "512MB",    "Sony",          NULL },
  { "0011",      "Q2 2014",    "Compute Module 1",    "1.0",    "512MB",    "Sony",          NULL },
  { "0012",      "Q4 2014",    "A+",                  "1.1",    "256MB",    "Sony",          NULL },
  { "0013",      "Q1 2015",    "B+",                  "1.2",    "512MB",    unk,             NULL },
  { "0014",      "Q2 2014",    "Compute Module 1",    "1.0",    "512MB",    "Embest",        NULL },
  { "0015",      unk,          "A+",                  "1.1",    "256MB/512MB",    "Embest",      NULL  },
  { "a01040",    unk,          "2 Model B",           "1.0",    "1GB",      "Sony",          "BCM2836" },
  { "a01041",    "Q1 2015",    "2 Model B",           "1.1",    "1GB",      "Sony",          "BCM2836" },
  { "a21041",    "Q1 2015",    "2 Model B",           "1.1",    "1GB",      "Embest",        "BCM2836" },
  { "a22042",    "Q3 2016",    "2 Model B",           "1.2",    "1GB",      "Embest",        "BCM2837" },  /* (with BCM2837) */
  { "900021",    "Q3 2016",    "A+",                  "1.1",    "512MB",    "Sony",          NULL },
  { "900032",    "Q2 2016?",    "B+",                 "1.2",    "512MB",    "Sony",          NULL },
  { "900092",    "Q4 2015",    "Zero",                "1.2",    "512MB",    "Sony",          NULL },
  { "900093",    "Q2 2016",    "Zero",                "1.3",    "512MB",    "Sony",          NULL },
  { "920093",    "Q4 2016?",   "Zero",                "1.3",    "512MB",    "Embest",        NULL },
  { "9000c1",    "Q1 2017",    "Zero W",              "1.1",    "512MB",    "Sony",          NULL },
  { "a02082",    "Q1 2016",    "3 Model B",           "1.2",    "1GB",      "Sony",          "BCM2837" },
  { "a020a0",    "Q1 2017",    "Compute Module 3 or CM3 Lite",  "1.0",    "1GB",    "Sony",          NULL },
  { "a22082",    "Q1 2016",    "3 Model B",           "1.2",    "1GB",      "Embest",        "BCM2837" },
  { "a32082",    "Q4 2016",    "3 Model B",           "1.2",    "1GB",      "Sony Japan",    NULL  },
  { NULL, NULL, NULL, NULL, NULL, NULL, NULL }
};

struct rpi_board {
    char *board_desc;

    /* from /proc/device-tree/model */
    char *dt_model;
    /* all from /proc/cpuinfo */
    char *soc, *revision, *serial;

    int overvolt; /* revision starts with 1000 (or maybe just 1) */

    /* all point into the rpi_boardinfo table -
     * no need to free */
    char *intro, *model, *pcb, *mem_spec, *mfg, *soc_spec;

    rpiz_fields *fields;
};

int rpi_board_check() {
    char *dtm;
    int ret = 0;
    dtm = get_dt_string("model");
    if (dtm) {
        ret = !(strstr(dtm, "Raspberry Pi") == NULL);
        free(dtm);
    }
    return ret;
}

/* return number of chars to skip */
static int ov_check(const char *r_code) {
    /* sources differ. prefix is either 1000... or just 1... */
    //if (strncmp(r, "1000", 4) == 0)
    //    return 4;
    if (strncmp(r_code, "1", 1) == 0)
        return 1;
    return 0;
}

static int code_match(const char* code0, const char* code1) {
    int c0, c1;
    if (code0 == NULL || code1 == NULL) return 0;
    c0 = strtol(code0, NULL, 16);
    c1 = strtol(code1, NULL, 16);
    if (c0 && c1)
        return (c0 == c1) ? 1 : 0;
    else
        return (strcmp(code0, code1) == 0) ? 1 : 0;
}

static int rpi_find_board(const char *r_code) {
    int i = 0;
    char *r = (char*)r_code;
    if (r_code == NULL)
        return 0;
    /* ignore the overvolt prefix */
    r += ov_check(r_code);
    while (rpi_boardinfo[i].value != NULL) {
        if (code_match(r, rpi_boardinfo[i].value))
            return i;

        i++;
    }
    return 0;
}

#ifndef PROC_CPUINFO
#define PROC_CPUINFO "/proc/cpuinfo"
#endif

#define CHECK_KV(k, v)  \
    if (strncmp(k, key, (strlen(k) < strlen(key)) ? strlen(k) : strlen(key)) == 0) { \
        if (b->v != NULL) free(b->v);                                                \
        b->v = malloc(strlen(value) + 1);                                      \
        if (b->v) strcpy(b->v, value); }

static int rpi_get_cpuinfo_data(rpi_board *b) {
    char *cpuinfo;
    kv_scan *kv; char *key, *value;

    cpuinfo = get_file_contents(PROC_CPUINFO);
    if (!cpuinfo) return 0;

    kv = kv_new(cpuinfo);
    if (kv) {
        while( kv_next(kv, &key, &value) ) {
            CHECK_KV("Revision", revision);
            CHECK_KV("Serial",   serial);
            CHECK_KV("Hardware", soc);
        }
    }
    kv_free(kv);
    free(cpuinfo);
    return 1;
}

static char* rpi_gen_board_name(int i) {
    char *ret = NULL;
    int l = 0;

    /* bounds check i */
    while(rpi_boardinfo[l].value != NULL) l++;
    if (i >= l) return NULL;

    ret = malloc(256);
    if (ret)
        snprintf(ret, 255, "Raspberry Pi %s Rev %s", rpi_boardinfo[i].model, rpi_boardinfo[i].pcb);
    return ret;
}

rpi_board *rpi_board_new() {
    int i = 0;
    rpi_board *s = malloc( sizeof(rpi_board) );
    if (s) {
        memset(s, 0, sizeof(*s));
        rpi_get_cpuinfo_data(s);

        i = rpi_find_board(s->revision);
        s->model = rpi_boardinfo[i].model;
        s->pcb = rpi_boardinfo[i].pcb;
        s->mfg = rpi_boardinfo[i].mfg;
        s->intro = rpi_boardinfo[i].intro;
        s->mem_spec = rpi_boardinfo[i].mem;
        s->soc_spec = rpi_boardinfo[i].soc;
        s->overvolt = 0;
        if (s->revision)
            if (ov_check(s->revision))
                s->overvolt = 1;

        s->dt_model = get_dt_string("model");
        if (i)
            s->board_desc = rpi_gen_board_name(i);
        else {
            if (s->dt_model)
                s->board_desc = s->dt_model;
            else
                s->board_desc = unk;
        }

        s->fields = NULL;
    }
    return s;
}

void rpi_board_free(rpi_board *s) {
    if (s) {
        free(s->soc);
        free(s->revision);
        free(s->serial);
        if (s->board_desc != s->dt_model && s->board_desc != unk)
            free(s->board_desc);
        free(s->dt_model);
        if (s->fields)
            fields_free(s->fields);
        free(s);
    }
}

const char *rpi_board_desc(rpi_board *s) {
    if (s)
        return s->board_desc;
    return NULL;
}

const char *rpi_board_rcode(rpi_board *s) {
    if (s)
        return s->revision;
    return NULL;
}

const char *rpi_board_serial(rpi_board *s) {
    if (s)
        return s->serial;
    return NULL;
}


const char *rpi_board_model(rpi_board *s) {
    if (s)
        return s->model;
    return NULL;
}

const char *rpi_board_rev(rpi_board *s) {
    if (s)
        return s->pcb;
    return NULL;
}

const char *rpi_board_intro(rpi_board *s) {
    if (s)
        return s->intro;
    return NULL;
}

const char *rpi_board_mfgby(rpi_board *s) {
    if (s)
        return s->mfg;
    return NULL;
}

const char *rpi_board_mem_spec(rpi_board *s) {
    if (s)
        return s->mem_spec;
    return NULL;
}

const char *rpi_board_soc(rpi_board *s) {
    if (s) {
        if (s->soc_spec)
            return s->soc_spec;
        else
            return s->soc;
    }
    return NULL;
}

int rpi_board_overvolt(rpi_board *s) {
    if (s)
        return s->overvolt;
    return 0;
}

float rpi_soc_temp() {
    char *tmp = NULL;
    float temp = 0.0f;
    tmp = get_file_contents("/sys/class/thermal/thermal_zone0/temp");
    if (tmp != NULL)
        temp = (float)atoi(tmp);
    if (temp)
        temp /= 1000.0f;
    free(tmp);
    return temp;
}

static char* rpi_soc_temp_str(void *s) {
    float t = rpi_soc_temp();
    char *buff;
    s = s; /* avoid a warning */
    buff = malloc(128);
    if (buff)
        snprintf(buff, 127, "%0.2f'C", t);
    return buff;
}

static char* rpi_board_overvolt_str(rpi_board *s) {
    char *buff = NULL;
    if (s) {
        buff = malloc(128);
        if (buff)
            snprintf(buff, 127, "%s", (rpi_board_overvolt(s)) ? "yes (warranty void!)" : "never" );
    }
    return buff;
}

#define ADDFIELD(t, l, o, n, f) fields_update_bytag(s->fields, t, l, o, n, (rpiz_fields_get_func)f, (void*)s)
#define ADDFIELDSTR(t, l, o, n, str) fields_update_bytag(s->fields, t, l, o, n, NULL, (void*)str)
rpiz_fields *rpi_board_fields(rpi_board *s) {
    if (s) {
        if (!s->fields) {
            /* first insert creates */
            s->fields =
            ADDFIELD("summary.board_name",  0, 0, "Board Name", rpi_board_desc );
            ADDFIELD("summary.rpi_temp",    1, 1, "SOC Temp",   rpi_soc_temp_str );
            ADDFIELD("board.rpi_name",      0, 0, "Model", rpi_board_desc );
            ADDFIELD("board.rpi_intro",     0, 0, "Introduction", rpi_board_intro );
            ADDFIELD("board.rpi_mfgby",     0, 0, "Manufacturer", rpi_board_mfgby );
            ADDFIELD("board.rpi_mem_spec",  0, 0, "Memory (spec)", rpi_board_mem_spec );
            if (s->soc_spec)
            ADDFIELDSTR("board.rpi_soc_spec",  0, 0, "SOC (spec)", s->soc_spec );
            ADDFIELDSTR("board.rpi_soc",    0, 0, "SOC (reported)", s->soc );
            ADDFIELD("board.rpi_rcode",     0, 0, "RCode", rpi_board_rcode );
            ADDFIELD("board.rpi_serial",    0, 0, "Serial Number", rpi_board_serial );
            ADDFIELD("board.rpi_overvolt",  0, 1, "Overvolt", rpi_board_overvolt_str );
            ADDFIELD("board.rpi_temp",      1, 1, "SOC Temp",   rpi_soc_temp_str );
        }
        return s->fields;
    }
    return NULL;
}

