#include "types.h"

#include "ksort.h"

// clang-format off
#include <htslib/sam.h>

#include <stddef.h>
#include <stdlib.h>

#include <limits>
// clang-format on

namespace kadayashi {

// sorting
#define generic_key(x) (x)
#define generic_cmp(x, y) ((x) < (y))
void rs_insertsort_ksu32(uint32_t *beg, uint32_t *end);
void rs_sort_ksu32(uint32_t *beg, uint32_t *end, int n_bits, int s);
void radix_sort_ksu32(uint32_t *beg, uint32_t *end);
KRADIX_SORT_INIT(ksu32, uint32_t, generic_key, 4);

void rs_insertsort_ksu64(uint64_t *beg, uint64_t *end);
void rs_sort_ksu64(uint64_t *beg, uint64_t *end, int n_bits, int s);
void radix_sort_ksu64(uint64_t *beg, uint64_t *end);
KRADIX_SORT_INIT(ksu64, uint64_t, generic_key, 8);

void ks_mergesort_kssu32(size_t n, uint32_t array[], uint32_t temp[]);
uint32_t ks_ksmall_kssu32(size_t n, uint32_t arr[], size_t kk);
void ks_shuffle_kssu32(size_t n, uint32_t a[]);
void ks_sample_kssu32(size_t n, size_t r, uint32_t a[]);
KSORT_INIT(kssu32, uint32_t, generic_cmp)

void ks_mergesort_kssu64(size_t n, uint64_t array[], uint64_t temp[]);
uint64_t ks_ksmall_kssu64(size_t n, uint64_t arr[], size_t kk);
void ks_shuffle_kssu64(size_t n, uint64_t a[]);
void ks_sample_kssu64(size_t n, size_t r, uint64_t a[]);
KSORT_INIT(kssu64, uint64_t, generic_cmp)

#define qa_t_cmp(x, y) ((x).pos < (y).pos)
void ks_mergesort_ksqa(size_t n, qa_t array[], qa_t temp[]);
qa_t ks_ksmall_ksqa(size_t n, qa_t arr[], size_t kk);
void ks_shuffle_ksqa(size_t n, qa_t a[]);
void ks_sample_ksqa(size_t n, size_t r, qa_t a[]);
KSORT_INIT(ksqa, qa_t, qa_t_cmp)

void destroy_vu32_v(vu32_v *h, int include_self) {
    for (size_t i = 0; i < h->n; i++) {
        kv_destroy(h->a[i]);
    }
    kv_destroy(*h);
    if (include_self) {
        free(h);
    }
}

ref_vars_t *init_ref_vars_t(const char *chrom, int bucket_l) {
    ref_vars_t *ret = (ref_vars_t *)calloc(1, sizeof(ref_vars_t));
    const size_t chrom_name_len = strlen(chrom) + 1;
    ret->chrom = (char *)calloc(chrom_name_len, 1);
    snprintf(ret->chrom, chrom_name_len, "%s", chrom);
    ret->bucket_l = bucket_l;
    kv_init(ret->poss);
    kv_init(ret->start_indices);
    return ret;
}
void destroy_ref_vars_t(ref_vars_t *h) {
    free(h->chrom);
    kv_destroy(h->poss);
    kv_destroy(h->start_indices);
    free(h);
}

// init_vg_t(): see vg_gen()
void destroy_vg_t(vg_t *vg) {
    free(vg->nodes);
    free(vg->edges);
    free(vg->next_link_is_broken);
    free(vg);
}

void init_vgnode_t(vgnode_t *h, uint32_t ID) {
    h->ID = ID;
    h->scores[0] = 0;
    h->scores[1] = 0;
    h->scores[2] = 0;
    h->scores[3] = 0;
    h->best_score_i = -1;
}

void init_read_t(read_t *h) {
    h->start_pos = UINT32_MAX;
    h->end_pos = UINT32_MAX;
    h->ID = UINT32_MAX;
    h->hp = HAPTAG_UNPHASED;
    h->strand = UINT8_MAX;
    h->vars = init_qa_v();
    h->left_clip_len = 0;
    h->right_clip_len = 0;
}

void destroy_read_t(read_t *h, int include_self) {
    destroy_qa_v(h->vars);
    if (include_self) {
        free(h);
    }
}

void destroy_chunk_t(chunk_t *h, int include_self) {
    for (size_t i = 0; i < h->reads.n; i++) {
        destroy_read_t(&h->reads.a[i], 0);
        hts_free(h->qnames[i]->s);
        free(h->qnames[i]);
        if (h->compat0) {
            free(h->compat0[i]);
        }
    }
    if (h->compat0) {
        free(h->compat0);
    }
    free(h->qnames);
    kv_destroy(h->reads);
    destroy_ta_v(h->varcalls);
    //htstri_ht_destroy(h->qname2ID);  // keys are alloced & freed in qnames
    kh_destroy(htstri_t, h->qname2ID);  // keys are alloced & freed in qnames
    if (include_self) {
        free(h);
    }
}

qa_v *init_qa_v(void) {
    qa_v *h = (qa_v *)calloc(1, sizeof(qa_v));
    kv_init(*h);
    return h;
}

void destroy_qa_v(qa_v *h) {
    for (size_t i = 0; i < h->n; i++) {
        kv_destroy(h->a[i].allele);
    }
    kv_destroy(*h);
    free(h);
}

void destroy_ta_t(ta_t *h, int include_self, int forced) {
    if (h->is_used || forced) {
        for (size_t i = 0; i < h->alleles.n; i++) {
            kv_destroy(h->alleles.a[i]);
            kv_destroy(h->allele2readIDs.a[i]);
        }
        kv_destroy(h->alleles);
        kv_destroy(h->allele2readIDs);
        kv_destroy(h->alleles_is_used);
    }
    if (include_self) {
        free(h);
    }
}

void destroy_ta_v(ta_v *h) {
    for (size_t i = 0; i < h->n; i++) {  // destroy each position
        destroy_ta_t(&h->a[i], 0, 0);
    }
    kv_destroy(*h);
    free(h);
}

static void add_dummy_qa_v(qa_v *h) {
    qa_t tmp;
    tmp.pos = UINT32_MAX;
    tmp.is_used = 0;
    tmp.allele_idx = UINT32_MAX;
    // do not init tmp.allele
    tmp.hp = std::numeric_limits<uint8_t>::max();
    kv_push(qa_t, *h, tmp);
}

void dummyexpand_qa_v(qa_v *h, int n) {
    for (size_t i = 0; i < h->n - n; i++) {
        add_dummy_qa_v(h);
    }
}

int sort_qa_v(qa_v *h, qa_v *buf) {
    if (h->n <= 1) {
        return 0;
    }
    if (h->n > buf->n) {
        dummyexpand_qa_v(buf, static_cast<int>(h->n));
    }
    ks_mergesort_ksqa(h->n, h->a, buf->a);
    return 1;
}

}  // namespace kadayashi
