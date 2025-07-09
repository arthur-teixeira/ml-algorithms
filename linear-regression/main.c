#include "../shared/da.h"
#include "../shared/data_structures.h"
#include <math.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct {
  size_t cap;
  size_t len;
  Vec *values;
} Samples;

// Dataset assumes both training and validation data are loaded in the same
// struct.
typedef struct {
  size_t dimensions;
  Samples samples;
  size_t training_cuttoff; // samples index from which validation data starts.
} Dataset;

Vec parse_line(char *line, size_t num_fields) {
  Vec v = new_vec(num_fields);
  char *tok = strtok(line, ",");
  size_t i = 0;
  while (tok != NULL) {
    double d;
    sscanf(tok, "%lf", &d);
    v.values[i++] = d;
    tok = strtok(NULL, ",");
  }

  return v;
}

size_t count_fields(char *line) {
  size_t count = 0;
  char *tok = strtok(line, ",");
  while (tok != NULL) {
    count++;
    tok = strtok(NULL, ",");
  }

  return count;
}

#define EXPECTED(s) s.values[s.size - 1]

Vec *vec_array(size_t size) { return calloc(sizeof(Vec), size); }

Dataset parse_wine_quality_dataset(char *filename) {
  FILE *f = fopen(filename, "rb");
  if (!f) {
    perror("fopen");
    exit(1);
  }

  char line[1024] = {0};
  fgets(line, sizeof(line), f);
  size_t num_fields = count_fields(line);

  Samples samples = {0};
  da_init((&samples), sizeof(Vec));

  do {
    memset(line, 0, sizeof(line));
    fgets(line, sizeof(line), f);
    Vec sample = parse_line(line, num_fields);
    da_append((&samples), sample);
  } while (strlen(line) > 0);

  return (Dataset){
      .samples = samples,
      // Last value is the dependent variable
      .dimensions = samples.values[0].size - 1,
      .training_cuttoff = samples.len / 2,
  };
}

void gradient_descent(Dataset d, double eta, double *w, double *b) {
  double d_w = 0.0, d_b = 0.0;

  size_t n = d.training_cuttoff;
  for (size_t i = 0; i < n; i++) {
    // MSE = 1/N Σ((y - (wx+ b))^2)
    // ∂d/∂w = 1/N Σ (2x(y - (wx + b))
    // ∂d/∂b = 1/N Σ 2(y - (wx + b)
    Vec sample = da_at(d.samples, i);
    double expected = EXPECTED(sample);

    double predicted = *w * sample.values[0] + *b;
    double t = (expected - predicted);
    d_w -= sample.values[0] * t;
    d_b -= t;
  }

  *w = *w - 2 * (eta / n) * d_w;
  *b = *b - 2 * (eta / n) * d_b;
}

double validate(Dataset d, double w, double b) {
  double mse_acc = 0.0f;
  for (size_t i = d.training_cuttoff; i < d.samples.len; i++) {
    Vec sample = da_at(d.samples, i);
    double expected = EXPECTED(sample);
    double predicted = w * sample.values[0] + b;
    mse_acc += fabs(expected - predicted) * fabs(expected - predicted);
  }
  return mse_acc / (d.samples.len - d.training_cuttoff);
}

#define EPOCHS 1000000

int main(void) {
  Dataset d = parse_wine_quality_dataset(
      "./linear-regression/data/winequality-red.csv");

  double w = 0.0, b = 0.0;
  double prev_loss = INFINITY;
  for (size_t i = 0; i < EPOCHS; i++) {
    gradient_descent(d, 0.001, &w, &b);

    if (i % 100 == 0) {
      double loss = validate(d, w, b);
      printf("Epoch %ld - Loss: %.9f\n", i, loss);
      if (fabs(loss - prev_loss) < 0.00001) {
        printf("Loss converged, stopping\n");
        break;
      }

      prev_loss = loss;
    }
  }
  return 0;
}
