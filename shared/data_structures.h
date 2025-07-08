#ifndef DATA_STRUCTURES_H
#define DATA_STRUCTURES_H
#include <assert.h>
#include <math.h>
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>

#define MAT_AT(m, i, j) (m).values[(i) * (m).cols + (j)]
#define VEC_AT(v, i) (v).values[(i)]
#define MAT_SIZE(m) (m).cols *(m).rows
#define FOREACH_VEC(v) for (size_t i = 0; i < (v).size; i++)

float rand_normal() { return ((float)rand() / (float)RAND_MAX); }
float rand_float_scaled(size_t fan_in, size_t fan_out) {
  // Xavier/Glorot Uniform
  float limit = sqrtf(6.0f / (fan_in + fan_out));
  float scale = 2.0f * limit;
  return (rand_normal())*scale - limit;
}

typedef struct {
  float *values;
  size_t rows;
  size_t cols;
} Mat;

typedef struct {
  float *values;
  size_t size;
} Vec;

float dot(Vec a, Vec b);
Vec matrix_vec_multiply(Mat w, Vec a);
Vec vec_sub(Vec a, Vec b);
Vec vec_add(Vec a, Vec b);
void free_mat(Mat m);
Mat new_mat(size_t rows, size_t cols);
void free_vec(Vec v);
Vec new_vec(size_t size);
Mat outer_product(Vec a, Vec b);

float sig(float i) { return 1 / (1 + exp(-i)); }
float d_sig(float i) { return sig(i) * (1 - sig(i)); }
Vec sig_vec(Vec a) {
  Vec s = new_vec(a.size);
  FOREACH_VEC(a) { VEC_AT(s, i) = sig(VEC_AT(a, i)); }
  return s;
}

Vec d_sig_vec(Vec a) {
  Vec d = new_vec(a.size);
  FOREACH_VEC(a) { VEC_AT(d, i) = sig(VEC_AT(a, i)); }
  return d;
}

Vec vectorize_image(uint8_t *image, size_t image_size) {
  Vec v = new_vec(image_size);
  for (size_t i = 0; i < image_size; i++) {
    VEC_AT(v, i) = (float)image[i];
  }

  return v;
}

Vec new_vec(size_t size) {
  Vec v = (Vec){
      .values = (float *)calloc(sizeof(float), size),
      .size = size,
  };
  assert(v.values != NULL);
  return v;
}

void free_vec(Vec v) { free(v.values); }

Vec rand_vec(size_t size, size_t fan_in, size_t fan_out) {
  Vec v = new_vec(size);
  FOREACH_VEC(v) { v.values[i] = rand_float_scaled(fan_in, fan_out); }
  return v;
}

Mat new_mat(size_t rows, size_t cols) {
  Mat mat = (Mat){
      .values = (float *)calloc(sizeof(float), rows * cols),
      .rows = rows,
      .cols = cols,
  };
  assert(mat.values != NULL);
  return mat;
}

void free_mat(Mat m) { free(m.values); }

Mat rand_matrix(size_t m, size_t n, size_t fan_in, size_t fan_out) {
  Mat mat = new_mat(m, n);
  for (size_t i = 0; i < m * n; i++) {
    mat.values[i] = rand_float_scaled(fan_in, fan_out);
  }
  return mat;
}

float dot(Vec a, Vec b) {
  assert(a.size == b.size);
  float acc = 0.0f;
  FOREACH_VEC(a) { acc += a.values[i] * b.values[i]; }
  return acc;
}

Vec vec_add(Vec a, Vec b) {
  Vec c = new_vec(a.size);
  FOREACH_VEC(a) { VEC_AT(c, i) = VEC_AT(a, i) + VEC_AT(b, i); }
  return c;
}

Vec vec_sub(Vec a, Vec b) {
  Vec c = new_vec(a.size);
  FOREACH_VEC(a) { VEC_AT(c, i) = VEC_AT(a, i) - VEC_AT(b, i); }
  return c;
}

Mat outer_product(Vec a, Vec b) {
  size_t rows = a.size;
  size_t cols = b.size;
  Mat result = new_mat(rows, cols);
  for (size_t i = 0; i < rows; i++) {
    for (size_t j = 0; j < cols; j++) {
      MAT_AT(result, i, j) = VEC_AT(a, i) * VEC_AT(b, j);
    }
  }

  return result;
}

#endif // DATA_STRUCTURES_H
