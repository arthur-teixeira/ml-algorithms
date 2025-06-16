#include <assert.h>
#include <math.h>
#include <time.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <raylib.h>
#include <raymath.h>

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
#define VEC_AT(v, i) (v).values[(i)]
#define MAT_SIZE(m) (m).cols *(m).rows
#define FOREACH_VEC(v) for (size_t i = 0; i < (v).size; i++)

typedef struct {
  float *values;
  size_t rows;
  size_t cols;
} Mat;

typedef struct {
  float *values;
  size_t size;
} Vec;

typedef struct {
  Vec data;
  Vec expected;
} Sample;

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

typedef Vec (*act)(Vec);

typedef struct {
  size_t neurons;
  act act;
  act dact;
} Layer;

typedef struct {
  size_t num_layers;
  Layer *layers;
  Vec *biases;
  Mat *weights;
} Network;

typedef struct {
  Vec *dnb;
  Mat *dnw;
} Backprop;

Backprop backprop(Network *net, Vec x, Vec y);
void free_backprop(Network *net, Backprop b);
void apply_gradient_mat(Mat *gradients, Mat *deltas, size_t num_mats);
void apply_gradient_vec(Vec *gradients, Vec *deltas, size_t num_vecs);
void free_mat_array(Network *net, Mat *v);
Mat *mat_array(Network *net, bool);
void free_vec_array(Network *net, Vec *v);
Vec *vec_array(Network *net, bool);
Sample **split_mini_batches(Sample *training_data, size_t n, size_t batch_size,
                            size_t *num_batches);
void shuffle(Sample *training_data, size_t n);
Vec feed_forward(Network *net, Vec a);
Vec matrix_vec_multiply(Mat w, Vec a);
Network new_network(size_t num_layers, Layer *layers);
Vec vec_sub(Vec a, Vec b);
Vec vec_add(Vec a, Vec b);
float dot(Vec a, Vec b);
void free_mat(Mat m);
Mat new_mat(size_t rows, size_t cols);
void free_vec(Vec v);
Vec new_vec(size_t size);
void update_batch(Network *net, Sample *batch, size_t batch_size, float eta);
Mat outer_product(Vec a, Vec b);

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

float rand_normal() { return ((float)rand() / (float)RAND_MAX); }
float rand_float_scaled(size_t fan_in, size_t fan_out) {
  // Xavier/Glorot Uniform
  float limit = sqrtf(6.0f / (fan_in + fan_out));
  float scale = 2.0f * limit;
  return (rand_normal())*scale - limit;
}

Vec expected(size_t label) {
  Vec v = new_vec(10);
  VEC_AT(v, label) = 1.0f;
  return v;
}

Vec vectorize_image(uint8_t *image, size_t image_size) {
  Vec v = new_vec(image_size);
  for (size_t i = 0; i < image_size; i++) {
    VEC_AT(v, i) = (float)image[i];
  }

  return v;
}

Sample *samples(Dataset d) {
  Sample *samples = calloc(d.image_count, sizeof(Sample));
  assert(samples != NULL);

  for (size_t i = 0; i < d.image_count; i++) {
    Sample s = (Sample){
        .data = vectorize_image(d.images[i], d.image_size),
        .expected = expected(d.labels[i]),
    };
    samples[i] = s;
  }

  return samples;
}

void free_sample(Sample s) {
  free_vec(s.data);
  free_vec(s.expected);
}

void free_samples(Sample *s, size_t num_samples) {
  for (size_t i = 0; i < num_samples; i++) {
    free_sample(s[i]);
  }

  free(s);
}

#define IMAGE 1000

Vec new_vec(size_t size) {
  Vec v = (Vec){
      .size = size,
      .values = calloc(sizeof(float), size),
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
      .rows = rows,
      .cols = cols,
      .values = calloc(sizeof(float), rows * cols),
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

Network new_network(size_t num_layers, Layer *layers) {
  Network net = (Network){
      .num_layers = num_layers,
      .layers = layers,
      .biases = calloc(sizeof(Vec), num_layers),
      .weights = calloc(sizeof(Mat), num_layers),
  };
  assert(net.biases != NULL);
  assert(net.weights != NULL);

  for (size_t i = 0; i < num_layers; i++) {
    size_t fan_in = i > 0 ? layers[i - 1].neurons : 0;
    size_t fan_out = layers[i].neurons;
    net.biases[i] = rand_vec(fan_out, fan_in, fan_out);
  }

  for (size_t i = 0; i < num_layers - 1; i++) {
    net.weights[i] = rand_matrix(layers[i + 1].neurons, layers[i].neurons,
                                 layers[i + 1].neurons, layers[i].neurons);
  }
  return net;
}

Vec matrix_vec_multiply(Mat w, Vec a) {
  assert(w.cols == a.size);
  Vec result = new_vec(w.rows);
  for (size_t i = 0; i < w.rows; i++) {
    for (size_t j = 0; j < w.cols; j++) {
      VEC_AT(result, i) += MAT_AT(w, i, j) * VEC_AT(a, j);
    }
  }

  return result;
}

Vec transposed_matrix_vec_multiply(Mat w, Vec a) {
  assert(w.rows == a.size);
  Vec result = new_vec(w.cols);
  for (size_t i = 0; i < w.cols; i++) {
    for (size_t j = 0; j < w.rows; j++) {
      VEC_AT(result, i) += MAT_AT(w, j, i) * VEC_AT(a, j);
    }
  }

  return result;
}

Vec feed_forward(Network *net, Vec a) {
  assert(a.size == net->layers[0].neurons);
  for (size_t i = 0; i < net->num_layers - 1; i++) {
    Mat w = net->weights[i];
    Vec b = net->biases[i];
    Vec wb = matrix_vec_multiply(w, a);
    Vec z = vec_add(wb, b);
    free_vec(wb);
    if (i > 0) {
      free_vec(a);
    }
    a = sig_vec(z);
    free_vec(z);
  }
  return a;
}

void shuffle(Sample *training_data, size_t n) {
  for (size_t i = n - 1; i > 0; i--) {
    size_t j = rand() % (i + 1);
    Sample t = training_data[i];
    training_data[i] = training_data[j];
    training_data[j] = t;
  }
}

Sample **split_mini_batches(Sample *training_data, size_t n, size_t batch_size,
                            size_t *num_batches) {
  *num_batches = n / batch_size;
  Sample **bs = calloc(sizeof(Sample *), *num_batches);
  for (size_t i = 0, j = 0; i < n; i += batch_size, j++) {
    bs[j] = &training_data[i];
  }

  return bs;
}

size_t eval_result(Vec output) {
  size_t max_i = 0;
  float max_val = 0.0f;
  for (size_t i = 0; i < output.size; i++) {
    if (output.values[i] > max_val) {
      max_i = (ssize_t)i;
      max_val = output.values[i];
    }
  }

  return max_i;
}

void evaluate(Network *net, size_t num_samples, Sample *validate_data) {
  size_t right = 0;
  size_t wrong = 0;
  for (size_t i = 0; i < num_samples; i++) {
    Vec result = feed_forward(net, validate_data[i].data);
    size_t ans = eval_result(result);
    size_t expected = eval_result(validate_data[i].expected);
    if (ans == expected) {
      right += 1;
    } else {
      wrong += 1;
    }
  }
  printf("Resultados: %ld/%ld corretos, %ld/%ld errados\n", right,
         num_samples, wrong, num_samples);
}

void gradient_descent(Network *net, Sample *training_data,
                      size_t training_samples, size_t epochs,
                      size_t mini_batch_size, float eta,
                      Sample *validation_data, size_t validation_samples) {
  for (size_t i = 0; i < epochs; i++) {
    shuffle(training_data, training_samples);
    size_t num_batches = 0;
    Sample **mini_batches = split_mini_batches(training_data, training_samples,
                                               mini_batch_size, &num_batches);
    for (size_t j = 0; j < num_batches; j++) {
      update_batch(net, mini_batches[j], mini_batch_size, eta);
    }
    printf("Epoca %ld - ", i);
    free(mini_batches);
    evaluate(net, validation_samples, validation_data);
  }
}

Vec *vec_array(Network *net, bool fill) {
  size_t num_vecs = net->num_layers - 1;
  Vec *a = calloc(sizeof(Vec), num_vecs);
  if (fill) {
    for (size_t i = 0; i < num_vecs; i++) {
      size_t vec_size = net->layers[i + 1].neurons;
      a[i] = new_vec(vec_size);
    }
  }
  return a;
}

void free_vec_array(Network *net, Vec *v) {
  for (size_t i = 0; i < net->num_layers - 1; i++) {
    free_vec(v[i]);
  }
  free(v);
}

Mat *mat_array(Network *net, bool fill) {
  size_t num_matrices = net->num_layers - 1;
  Mat *a = calloc(sizeof(Mat), num_matrices);
  if (fill) {
    for (size_t i = 0; i < num_matrices; i++) {
      size_t mat_rows = net->layers[i + 1].neurons;
      size_t mat_cols = net->layers[i].neurons;
      a[i] = new_mat(mat_rows, mat_cols);
    }
  }

  return a;
}

void free_mat_array(Network *net, Mat *v) {
  for (size_t i = 0; i < net->num_layers - 1; i++) {
    free_mat(v[i]);
  }
  free(v);
}

void update_batch(Network *net, Sample *batch, size_t batch_size, float eta) {
  Vec *gradient_b = vec_array(net, true);
  Mat *gradient_w = mat_array(net, true);

  for (size_t i = 0; i < batch_size; i++) {
    Vec x = batch[i].data;
    Vec y = batch[i].expected;
    Backprop b = backprop(net, x, y);
    apply_gradient_mat(gradient_w, b.dnw, net->num_layers - 1);
    apply_gradient_vec(gradient_b, b.dnb, net->num_layers - 1);
    free_backprop(net, b);
  }

  for (size_t i = 0; i < net->num_layers - 1; i++) {
    for (size_t j = 0; j < gradient_b[i].size; j++) {
      VEC_AT(net->biases[i], j) -=
          (eta / batch_size) * VEC_AT(gradient_b[i], j);
    }
  }

  for (size_t i = 0; i < net->num_layers - 1; i++) {
    for (size_t j = 0; j < MAT_SIZE(gradient_w[i]); j++) {
      VEC_AT(net->weights[i], j) -=
          (eta / batch_size) * VEC_AT(gradient_w[i], j);
    }
  }

  free_vec_array(net, gradient_b);
  free_mat_array(net, gradient_w);
}

void apply_gradient_vec(Vec *gradients, Vec *deltas, size_t num_vecs) {
  for (size_t i = 0; i < num_vecs; i++) {
    assert(gradients[i].size == deltas[i].size);
    for (size_t j = 0; j < gradients[i].size; j++) {
      VEC_AT(gradients[i], j) += VEC_AT(deltas[i], j);
    }
  }
}

void apply_gradient_mat(Mat *gradients, Mat *deltas, size_t num_mats) {
  for (size_t i = 0; i < num_mats; i++) {
    assert(gradients[i].cols == deltas[i].cols);
    assert(gradients[i].rows == deltas[i].rows);
    for (size_t j = 0; j < MAT_SIZE(gradients[i]); j++) {
      VEC_AT(gradients[i], j) += VEC_AT(deltas[i], j);
    }
  }
}

Vec cost_derivative(Vec activation, Vec expected) {
  return vec_sub(activation, expected);
}

Vec hadamard(Vec a, Vec b) {
  assert(a.size == b.size);
  Vec c = new_vec(a.size);
  FOREACH_VEC(a) { VEC_AT(c, i) = VEC_AT(a, i) * VEC_AT(b, i); }
  return c;
}

Vec softmax(Vec z) {
  float max = eval_result(z);
  float sum = 0.0;

  Vec result = new_vec(z.size);
  for (size_t i = 0; i < z.size; i++) {
    result.values[i] = expf(z.values[i] - max);
    sum += result.values[i];
  }

  for (size_t i = 0; i < z.size; i++) {
    result.values[i] /= sum;
  }

  return result;
}

Backprop backprop(Network *net, Vec x, Vec y) {
  Vec *gradient_b = vec_array(net, false);
  Mat *gradient_w = mat_array(net, false);

  size_t n = net->num_layers - 1;

  Vec activation = x;
  Vec activations[n + 1];
  activations[0] = x;
  Vec zs[n];

  for (size_t i = 0; i < n; i++) {
    Mat w = net->weights[i];
    Vec b = net->biases[i];
    Vec product = matrix_vec_multiply(w, activation);
    Vec z = vec_add(product, b);
    zs[i] = z;
    free_vec(product);
    Layer l = net->layers[i];
    activation = l.act(z);
    activations[1 + i] = activation;
  }

  Vec dcost = cost_derivative(activations[n], y);
  Layer out = net->layers[n];
  Vec dact = out.dact(zs[n - 1]);
  Vec delta = hadamard(dcost, dact);
  free_vec(dcost);
  free_vec(dact);

  gradient_b[n - 1] = delta;
  gradient_w[n - 1] = outer_product(delta, activations[n - 1]);

  for (ssize_t i = n - 2; i >= 0; i--) {
    Vec z = zs[i];
    Vec dact = d_sig_vec(z);
    Vec dcost = transposed_matrix_vec_multiply(net->weights[i + 1], delta);
    delta = hadamard(dcost, dact);
    free_vec(dcost);
    free_vec(dact);

    gradient_b[i] = delta;
    gradient_w[i] = outer_product(delta, activations[i]);
  }

  for (size_t i = 0; i < n; i++) {
    if (i > 0) {
      free_vec(activations[i]);
    }
    free_vec(zs[i]);
  }
  return (Backprop){
      .dnb = gradient_b,
      .dnw = gradient_w,
  };
}

void free_backprop(Network *net, Backprop b) {
  free_vec_array(net, b.dnb);
  free_mat_array(net, b.dnw);
}

#define ARRAY_LEN(xs) sizeof(xs) / sizeof(xs[0])

#define SCREEN_WIDTH 800
#define SCREEN_HEIGHT 600

#define IMAGE_SIZE 28
#define CANVAS_SIZE (420)
#define CELL_SIZE (CANVAS_SIZE / IMAGE_SIZE)

float canvas[IMAGE_SIZE * IMAGE_SIZE];
#define CANVAS_AT(i, j) (canvas)[(i) * IMAGE_SIZE + (j)]
#define CANVAS_X 20
#define CANVAS_Y 20
Vector2 canvas_pos = (Vector2){CANVAS_X, CANVAS_Y};

float blur_kernel[3][3] = {{1 / 16.0f, 2 / 16.0f, 1 / 16.0f},
                           {2 / 16.0f, 1, 2 / 16.0f},
                           {1 / 16.0f, 2 / 16.0f, 1 / 16.0f}};

void init_canvas() {
  for (int i = 0; i < IMAGE_SIZE; i++) {
    for (int j = 0; j < IMAGE_SIZE; j++) {
      int dx = j * CELL_SIZE;
      int dy = i * CELL_SIZE;
      Color c = {255, 255, 255, 0};
      c.a = CANVAS_AT(i, j);
      DrawRectangle(CANVAS_X + dx, CANVAS_Y + dy, CELL_SIZE, CELL_SIZE, c);
    }
  }

  for (int i = 0; i <= IMAGE_SIZE; i++) {
    int dx = i * CELL_SIZE;
    DrawLine(CANVAS_X + dx, CANVAS_Y, CANVAS_X + dx, CANVAS_Y + CANVAS_SIZE,
             GRAY);
    DrawLine(CANVAS_X, CANVAS_Y + dx, CANVAS_X + CANVAS_SIZE, CANVAS_Y + dx,
             GRAY);
  }
}

void paint_cell_with_blur(int x, int y) {
  for (int dy = -1; dy <= 1; dy++) {
    for (int dx = -1; dx <= 1; dx++) {
      int nx = x + dx;
      int ny = y + dy;
      if (nx >= 0 && nx < IMAGE_SIZE && ny >= 0 && ny < IMAGE_SIZE) {
        float weight = blur_kernel[dy + 1][dx + 1];
        float intensity = 3 * weight;
        float *pixel = &CANVAS_AT(ny, nx);
        *pixel = fminf(255.0f, *pixel + intensity);
      }
    }
  }
}

void capture_mouse_events() {
  if (IsMouseButtonDown(MOUSE_BUTTON_LEFT)) {
    Vector2 pos = GetMousePosition();
    Vector2 canvas_p = Vector2Subtract(pos, canvas_pos);
    int cell_x = (int)canvas_p.x / CELL_SIZE;
    int cell_y = (int)canvas_p.y / CELL_SIZE;

    if (cell_x < IMAGE_SIZE && cell_y < IMAGE_SIZE) {
      paint_cell_with_blur(cell_x, cell_y);
    }
  }
}

#define EPOCHS 30

int main() {
  memset(canvas, 0, sizeof(canvas));
  srand(time(NULL));
  Dataset data = load_mnist_dataset("./data/lg/train-images.idx3-ubyte",
                                    "./data/lg/train-labels.idx1-ubyte");
  Sample *test_samples = samples(data);

  Dataset validate = load_mnist_dataset("./data/lg/t10k-images.idx3-ubyte",
                                        "./data/lg/t10k-labels.idx1-ubyte");
  Sample *validate_samples = samples(validate);

  Layer layers[] = {
      {.neurons = 784, .act = sig_vec, .dact = d_sig_vec},
      {.neurons = 100, .act = sig_vec, .dact = d_sig_vec},
      {.neurons = 10, .act = sig_vec, .dact = d_sig_vec},
  };
  Network net = new_network(ARRAY_LEN(layers), layers);
  printf("Epoca -1 - ");
  evaluate(&net, validate.image_count, validate_samples);
  gradient_descent(&net, test_samples, data.image_count, EPOCHS, 10, 3.0,
                   validate_samples, validate.image_count);

  Vec canvas_vec = (Vec){
      .size = IMAGE_SIZE * IMAGE_SIZE,
      .values = canvas,
  };

  InitWindow(SCREEN_WIDTH, SCREEN_HEIGHT, "MNIST");

  int last_result = -1;
  while (!WindowShouldClose()) {
    capture_mouse_events();
    BeginDrawing();
    ClearBackground(BLACK);
    if (IsKeyPressed(KEY_T)) {
      Vec output = feed_forward(&net, canvas_vec);
      last_result = eval_result(output);
    }
    if (last_result >= 0) {
      int text_x = CANVAS_X + CANVAS_SIZE + 20;
      int text_y = CANVAS_Y;
      DrawText("Resultado:", text_x, text_y, 20, WHITE);
      char result_text[32];
      snprintf(result_text, sizeof(result_text), "%d", last_result);
      DrawText(result_text, text_x, text_y + 30, 40, YELLOW);
    }

    if (IsKeyPressed(KEY_SPACE)) {
      memset(canvas, 0, sizeof(canvas));
      last_result = -1;
    }
    init_canvas();
    EndDrawing();
  }

  return 0;
}
