#include "kadayashi_utils.h"

#include "htslib/khash_str2int.h"
#include "kstring.h"
#include "kvec.h"

#include <assert.h>
#include <limits.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define READLINE_BUF_LEN 1024
typedef kvec_t(char) vchar_t;
typedef kvec_t(uint64_t) vu64_t;

typedef struct {
    uint32_t ID;
    uint32_t start, end;
    uint32_t storage_size;
    uint32_t storage_n;  // aka # reads tagged.
                         // double negate this to check if chunk is phased
    uint64_t start_pos_in_bin;
} lite_chunk_info_t;
typedef kvec_t(lite_chunk_info_t) lite_chunk_info_v;

int pushusbsfwd(char *buf, int buf_l, int start) {
    memmove(buf, buf + start, buf_l - start);
    return buf_l - start;
}

void write_binary_given_tsv(char *fn_tsv, char *fn_bin) {
    // selfnote for first serialize impl:
    // tsv must have been written sorted: it iterated through
    // references, and slided from left to right in each reference,
    // and the chunkID is 0-index integer that is continuously incremental
    // through this process.

    FILE *fp_tsv = fopen(fn_tsv, "r");
    assert(fp_tsv);
    FILE *fp_bin = fopen(fn_bin, "wb");
    assert(fp_bin);

    // for indexing
    vchar_t refnames;
    kv_init(refnames);
    uint32_t tot_refs = 0;
    char *last_refname = (char *)calloc(256, 1);
    int last_refname_n = 0;
    vu64_t ref2chunkIDrange;
    kv_init(ref2chunkIDrange);
    lite_chunk_info_v chunkinfos;
    kv_init(chunkinfos);

    int m = READLINE_BUF_LEN;
    int m_tmp = m;
    int n = 0;
    char *buffer = (char *)malloc(m);
    int offset = 0;

    int n_read;
    char *tok;
    int tok_l;
    uint64_t chunkID = 0;
    for (int pass = 0; pass < 2; pass++) {
        fseek(fp_tsv, 0, SEEK_SET);
        n = offset = 0;
        while ((n_read = fread(buffer + offset, 1, m - offset, fp_tsv)) > 0) {
            int start = 0;
            for (int i = 0; i < offset + n_read; i++) {
                if (buffer[i] == '\n') {
                    buffer[i] = 0;
                    char *line = buffer + start;

                    // parse the line
                    if (line[0] == 'C' && pass == 0) {  // encountered a new chunk block
                        kv_push(lite_chunk_info_t, chunkinfos,
                                ((lite_chunk_info_t){.ID = chunkID,
                                                     .start = UINT32_MAX,
                                                     .end = UINT32_MAX,
                                                     .storage_size = 0,
                                                     .storage_n = 0,
                                                     .start_pos_in_bin = UINT64_MAX}));
                        tok = strtok(line, "\t");
                        int i_col = 0;
                        while (tok) {
                            tok = strtok(NULL, "\t");
                            if (!tok) {
                                break;
                            }
                            i_col++;
                            if (i_col == 2) {  // refname
                                assert(strlen(tok) <= 255);
                                if (last_refname_n == 0 || strcmp(tok, last_refname) != 0) {
                                    sprintf(last_refname, "%s", tok);
                                    last_refname_n = strlen(tok);
                                    kv_push(char, refnames,
                                            (char)strlen(tok));  // lead a refname with its length
                                    for (int tmpi = 0; tmpi < strlen(tok);
                                         tmpi++) {  // and doesn't end with null
                                        kv_push(char, refnames, tok[tmpi]);
                                    }
                                    kv_push(uint64_t, ref2chunkIDrange,
                                            chunkID << 32 | (chunkID + 1));
                                    tot_refs++;
                                } else {
                                    ref2chunkIDrange.a[ref2chunkIDrange.n - 1]++;
                                }
                            } else if (i_col == 3) {  // ref start
                                chunkinfos.a[chunkinfos.n - 1].start = strtoul(tok, NULL, 10);
                            } else if (i_col == 4) {  // ref end
                                chunkinfos.a[chunkinfos.n - 1].end = strtoul(tok, NULL, 10);
                            }
                        }
                        chunkID++;
                    } else if (line[0] == 'R' && pass == 0) {  // collect size of storages
                        tok = strtok(line, "\t");
                        int i_col = 0;
                        while (tok) {
                            tok = strtok(NULL, "\t");
                            if (!tok) {
                                break;
                            }
                            i_col++;
                            if (i_col == 2) {  // read name
                                if (strlen(tok) > UINT8_MAX) {
                                    fprintf(stderr, "[E::%s] read name too long (l=%d)\n", __func__,
                                            (int)strlen(tok));
                                    exit(1);
                                }
                                int size = sizeof(uint8_t) /*length of qname*/ +
                                           strlen(tok) /*qname string, no null term*/ +
                                           1 /*uint8_t haptag*/;
                                chunkinfos.a[chunkinfos.n - 1].storage_n += 1;
                                chunkinfos.a[chunkinfos.n - 1].storage_size += size;
                                //fprintf(stderr, "nya chunk=%d storage_n=%d\n", chunkID,
                                //    chunkinfos.a[chunkinfos.n-1].storage_n );
                                break;
                            }
                        }
                    } else if (line[0] == 'R' && pass == 1) {  // write chunk contents to storage
                        tok = strtok(line, "\t");
                        int i_col = 0;
                        while (tok) {
                            tok = strtok(NULL, "\t");
                            if (!tok) {
                                break;
                            }
                            i_col++;
                            if (i_col == 2) {  // read name
                                uint8_t qname_l = strlen(tok);
                                fwrite(&qname_l, 1, 1, fp_bin);
                                fwrite(tok, 1, qname_l, fp_bin);
                            } else if (i_col == 3) {  // haptag
                                uint8_t haptag = atoi(tok);
                                fwrite(&haptag, 1, 1, fp_bin);
                                break;
                            }
                        }
                    }

                    // step in the lines buffer
                    start = i + 1;
                }
            }
            if (start != 0) {
                offset = pushusbsfwd(buffer, m, start);
            } else {  // did not see a full line
                offset = m;
                m = m + (m >> 1);
                buffer = (char *)realloc(buffer, m);
            }
        }

        // if nothing is collected, return now
        if (chunkID == 0) {
            fprintf(stderr, "[W::%s] no chunck collected, nothing to write\n", __func__);
            goto cleanup;
        }

        // if first pass is done, write the header part of the binary file now
        // and let second pass write storages.
        if (pass == 0) {
            fprintf(stderr,
                    "[dbg::%s] to write header: %d refs, %d chunks (sancheck: chunkinfo buf length "
                    "%d)\n",
                    __func__, tot_refs, (int)chunkID, (int)chunkinfos.n);
            uint32_t header_offset1 =
                    sizeof(uint32_t) + refnames.n  // tot_refs and refnames_s_concat
                    +
                    tot_refs *
                            (sizeof(uint64_t) +
                             sizeof(uint32_t));  // for each ref, start as-in-file position of chunk interval info, and how many intervals are there
            uint32_t header_offset2 =
                    chunkID *
                    (sizeof(uint32_t) * 2 + sizeof(uint64_t) +
                     sizeof(uint32_t));  // chunk interval infos: ref_start, ref_end, start as-in-file position of the chunk, and how many entries (qname-haptag pairs) in chunk

            // collect start_pos_in_bin
            chunkinfos.a[0].start_pos_in_bin = header_offset1 + header_offset2;
            for (int ic = 1; ic < chunkID; ic++) {
                chunkinfos.a[ic].start_pos_in_bin =
                        chunkinfos.a[ic - 1].start_pos_in_bin + chunkinfos.a[ic - 1].storage_size;
            }

            // write headers
            //(refnames)
            fwrite(&tot_refs, sizeof(uint32_t), 1, fp_bin);
            fwrite(refnames.a, refnames.n, 1, fp_bin);
            // (how to jump to chunk interval infos for each reference)
            for (int i_ref = 0; i_ref < tot_refs; i_ref++) {
                uint32_t chunkID_start = ref2chunkIDrange.a[i_ref] >> 32;
                uint64_t chunkID_start_infile =
                        header_offset1 +
                        chunkID_start * (sizeof(uint32_t) * 3 + sizeof(uint64_t) * 1);
                uint32_t chunkID_n = (uint32_t)ref2chunkIDrange.a[i_ref] - chunkID_start;
                fwrite(&chunkID_start_infile, sizeof(uint64_t), 1, fp_bin);
                fwrite(&chunkID_n, sizeof(uint32_t), 1, fp_bin);
            }
            fflush(fp_bin);
            assert(ftell(fp_bin) == (header_offset1));
            // (chunk interval infos: ref_start, ref_end, start_pos_in_bin, storage_n)
            for (int ic = 0; ic < chunkinfos.n; ic++) {
                uint32_t ref_s = chunkinfos.a[ic].start;
                uint32_t ref_e = chunkinfos.a[ic].end;
                uint64_t pos_infile = chunkinfos.a[ic].start_pos_in_bin;
                uint32_t l_inchunk = chunkinfos.a[ic].storage_n;  // count, not bytes
                fwrite(&ref_s, sizeof(uint32_t), 1, fp_bin);
                fwrite(&ref_e, sizeof(uint32_t), 1, fp_bin);
                fwrite(&pos_infile, sizeof(uint64_t), 1, fp_bin);
                fwrite(&l_inchunk, sizeof(uint32_t), 1, fp_bin);
            }
            fflush(fp_bin);
            assert(ftell(fp_bin) == (header_offset1 + header_offset2));
        }  // end of header write
    }
    fprintf(stderr, "[M::%s] wrote bin file\n", __func__);

cleanup:
    free(buffer);
    fclose(fp_tsv);
    fclose(fp_bin);

    kv_destroy(ref2chunkIDrange);
    kv_destroy(chunkinfos);
    kv_destroy(refnames);
    free(last_refname);
    return;
}

void *query_bin_file_get_qname2tag(char *fn_bin,
                                   char *chrom,
                                   uint32_t ref_start,
                                   uint32_t ref_end) {
    int silent = 0;
    int debug_print = 0;
    khash_t(str2int) *qname2tag = khash_str2int_init();
    assert(fn_bin);
    FILE *fp = fopen(fn_bin, "rb");

    uint32_t n_ref;
    fread(&n_ref, sizeof(uint32_t), 1, fp);

    // get chrom index
    int refname_m = 16;
    char *refname = (char *)calloc(refname_m, 1);
    int ref_i = -1;
    for (int i = 0; i < n_ref; i++) {
        uint8_t tn_l;
        fread(&tn_l, 1, 1, fp);
        if (ref_i < 0) {  // haven't found the chrom yet
            if (tn_l >= refname_m) {
                refname_m = tn_l + 1;
                refname = (char *)realloc(refname, refname_m);
            }
            fread(refname, 1, tn_l, fp);
            refname[tn_l] = 0;
            if (strcmp(refname, chrom) == 0) {  // done; don't break
                if (debug_print) {
                    fprintf(stderr, "[dbg::%s] found ref %s\n", __func__, refname);
                }
                ref_i = i;
            }
        } else {
            fseek(fp, tn_l, SEEK_CUR);
        }
    }
    if (ref_i >= 0) {
        int found = 0;
        uint64_t pos_intervals_start;
        uint32_t n_intervals;
        fseek(fp, ref_i * (sizeof(uint64_t) + sizeof(uint32_t)), SEEK_CUR);

        fread(&pos_intervals_start, sizeof(uint64_t), 1, fp);
        fread(&n_intervals, sizeof(uint32_t), 1, fp);
        fseek(fp, pos_intervals_start, SEEK_SET);

        // search to find the ID of a fulfilling chunk.
        uint32_t start, end, n_reads;
        uint64_t pos_chunk_start;
        uint32_t best_ovlp_len = 0;  // will check all overlapping intervals and take
                                     // the one with largest ovlp. Tie break is arbitrary
                                     // though stable wrt the bin file.
        uint64_t best_ovlp_chunk_start = 0;
        uint32_t best_ovlp_nreads = 0;
        uint32_t best_ovlp_start = 0, best_ovlp_end = 0;
        for (int i = 0; i < n_intervals; i++) {
            fread(&start, sizeof(uint32_t), 1, fp);
            fread(&end, sizeof(uint32_t), 1, fp);
            fread(&pos_chunk_start, sizeof(uint64_t), 1, fp);
            fread(&n_reads, sizeof(uint32_t), 1, fp);
            if (ref_end > start && ref_start < end) {
                if (debug_print) {
                    fprintf(stderr, "[dbg::%s] checking %d-%d\n", __func__, start, end);
                }
                uint64_t l = 0;
                if (start < ref_start) {
                    l = end > ref_end ? ref_end - ref_start : end - ref_start;
                } else {
                    l = end > ref_end ? ref_end - start : end - start;
                }
                if (l > best_ovlp_len) {
                    if (debug_print) {
                        fprintf(stderr,
                                "[dbg::%s] update best hit to: %d-%d with %d reads (l: %d => %d)\n",
                                __func__, start, end, n_reads, (int)best_ovlp_len, (int)l);
                    }
                    best_ovlp_len = l;
                    best_ovlp_chunk_start = pos_chunk_start;
                    best_ovlp_nreads = n_reads;
                    best_ovlp_start = start;
                    best_ovlp_end = end;
                } else {
                    if (debug_print) {
                        fprintf(stderr, "[dbg::%s] hit (worse): %d-%d with %d reads\n", __func__,
                                start, end, n_reads);
                    }
                }
            } else if (start > ref_end) {
                break;
            }
        }
        if (best_ovlp_chunk_start > 0) {
            fseek(fp, best_ovlp_chunk_start, SEEK_SET);
            fprintf(stderr, "[M::%s] use interval %d-%d\n", __func__, best_ovlp_start,
                    best_ovlp_end);

            // read the chunk
            uint8_t qn_l, haptag;
            char qn[255];
            for (int j = 0; j < best_ovlp_nreads; j++) {
                fread(&qn_l, 1, 1, fp);
                fread(qn, qn_l, 1, fp);
                qn[qn_l] = 0;
                fread(&haptag, 1, 1, fp);
                int absent;
                char *qname_key = (char *)calloc(qn_l + 1, 1);
                sprintf(qname_key, "%s", qn);
                khint_t key = kh_put(str2int, qname2tag, qname_key, &absent);  //selfnote: khash.h
                if (absent) {
                    // selfnote: this makes another put call, but kadayashi uses khashl.h
                    // which has a different kh_val() macro than khash.h.
                    khash_str2int_set(qname2tag, qname_key,
                                      haptag);  //kh_val(qname2tag, key) = haptag;
                    if (debug_print > 1) {
                        fprintf(stderr, "[dbg::%s] insert qn %s tag %d\n", __func__, qname_key,
                                haptag);
                    }
                } else {
                    if (debug_print) {
                        fprintf(stderr, "[dbg::%s] qn %s already seen\n", __func__, qname_key);
                    }
                    free(qname_key);
                }
            }
            found = 1;
            //break;  // EDITME
        }
        if (!found) {
            fprintf(stderr, "[W::%s] ref found, but requested interval not found (%s:%d-%d)\n",
                    __func__, chrom, ref_start, ref_end);
        }
    } else {
        if (!silent) {
            fprintf(stderr, "[W::%s] ref %s not found in bin's header\n", __func__, chrom);
        }
    }

    free(refname);
    fclose(fp);
    return (void *)qname2tag;
}
