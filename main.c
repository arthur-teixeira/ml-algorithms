#include <assert.h>
#include <math.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int reverse_int(int i) {
  int c1 = i & 255;
  int c2 = (i >> 8) & 255;
  int c3 = (i >> 16) & 255;
  int c4 = (i >> 24) & 255;
  return (c1 << 24) + (c2 << 16) + (c3 << 8) + c4;
}

typedef struct {
  uint8_t **images;
  uint8_t *labels;
  size_t label_count;
  size_t rows;
  size_t cols;
  size_t image_size;
  size_t image_count;
} Dataset;

#define MAT_AT(m, i, j) (m).values[(i) * (m).cols + (j)]

typedef struct {
  float *values;
  size_t rows;
  size_t cols;
} Mat;

typedef struct {
  float *values;
  size_t size;
} Vec;

uint8_t **load_mnist_images(const char *filename, int *image_count,
                            int *image_size, int *rows, int *cols) {
  FILE *fp = fopen(filename, "rb");
  if (!fp) {
    perror("could not open image");
    exit(1);
  }

  int magic = 0;
  fread(&magic, sizeof(int), 1, fp);
  fread(image_count, sizeof(int), 1, fp);
  fread(rows, sizeof(int), 1, fp);
  fread(cols, sizeof(int), 1, fp);

  *image_count = reverse_int(*image_count);
  *rows = reverse_int(*rows);
  *cols = reverse_int(*cols);
  *image_size = *rows * *cols;

  uint8_t **images = calloc(sizeof(void *), *image_count);
  for (int i = 0; i < *image_count; i++) {
    images[i] = calloc(*image_size, 1);
    fread(images[i], sizeof(uint8_t), *image_size, fp);
  }

  fclose(fp);

  return images;
}

uint8_t *load_mnist_labels(const char *filename, int *label_count) {
  FILE *fp = fopen(filename, "rb");
  if (!fp) {
    perror("could not open labels");
    exit(1);
  }

  int magic = 0;

  fread(&magic, sizeof(int), 1, fp);
  fread(label_count, sizeof(int), 1, fp);
  magic = reverse_int(magic);
  *label_count = reverse_int(*label_count);

  uint8_t *labels = calloc(*label_count, sizeof(uint8_t));
  fread(labels, sizeof(uint8_t), *label_count, fp);

  fclose(fp);
  return labels;
}

const char *brightness =
    "$@B%8&WM#*oahkbdpqwmZO0QLCJUYXzcvunxrjft/\\|()1{}[]?-_+~<>i!lI;:,\"^`'.";

Dataset load_mnist_dataset(const char *images_filename,
                           const char *labels_filename) {
  int image_count, image_size, rows, cols, label_count = 0;

  uint8_t **images = load_mnist_images(images_filename, &image_count,
                                       &image_size, &rows, &cols);
  uint8_t *labels = load_mnist_labels(labels_filename, &label_count);
  assert(image_count == label_count);

  return (Dataset){
      .images = images,
      .image_size = image_size,
      .image_count = image_count,
      .cols = cols,
      .rows = rows,
      .labels = labels,
      .label_count = label_count,
  };
}

inline float sig(float i) { return 1 / (1 + exp(-i)); }

float rand_float() { return (float)rand() / (float)RAND_MAX; }

#define IMAGE 1000

typedef struct {
  size_t num_layers;
  size_t *layer_sizes;
  Vec *biases;
  Mat *weights;
} Network;

Vec rand_vec(size_t size) {
  Vec v = (Vec){
      .size = size,
      .values = calloc(sizeof(float), size),
  };
  assert(v.values != NULL);

  for (size_t i = 0; i < size; i++) {
    v.values[i] = rand_float();
  }

  return v;
}

Mat rand_matrix(size_t m, size_t n) {
  Mat mat = (Mat){
      .rows = m,
      .cols = n,
      .values = calloc(sizeof(float), m * n),
  };
  assert(mat.values != NULL);

  for (size_t i = 0; i < m * n; i++) {
    mat.values[i] = rand_float();
  }

  return mat;
}

Network new_network(size_t num_layers, size_t *layer_sizes) {
  Network net = (Network){
      .num_layers = num_layers,
      .layer_sizes = layer_sizes,
      .biases = calloc(sizeof(Vec), num_layers),
      .weights = calloc(sizeof(Mat), num_layers),
  };
  assert(net.biases != NULL);
  assert(net.weights != NULL);

  for (size_t i = 0; i < num_layers; i++) {
    net.biases[i] = rand_vec(layer_sizes[i]);
  }

  for (size_t i = 0; i < num_layers - 1; i++) {
    net.weights[i] = rand_matrix(layer_sizes[i+1], layer_sizes[i]);
  }

  return net;
}

int main() {
  Dataset data = load_mnist_dataset("./data/lg/train-images.idx3-ubyte",
                                    "./data/lg/train-labels.idx1-ubyte");
  printf("Loaded %ld images and %ld labels\n", data.image_count,
         data.label_count);
  printf("Label for first image is %d", data.labels[IMAGE]);
  for (int i = 0; i < data.image_size; i++) {
    if (i % data.cols == 0)
      printf("\n");
    uint8_t intensity = data.images[IMAGE][i];
    printf("%c", brightness[intensity % 70]);
  }
  printf("\n");

  return 0;
}
