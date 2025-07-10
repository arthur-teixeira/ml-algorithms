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

Vec parse_line(char *line, size_t num_fields, char *delim) {
  Vec v = new_vec(num_fields);
  char *tok = strtok(line, delim);
  size_t i = 0;
  while (tok != NULL) {
    if (strlen(tok) > 0) {
      double d;
      sscanf(tok, "%lf", &d);
      v.values[i++] = d;
    }
    tok = strtok(NULL, delim);
  }

  return v;
}

size_t count_fields(char *line, char *delim) {
  size_t count = 0;
  char *tok = strtok(line, delim);
  while (tok != NULL) {
    if (strlen(tok) > 0) {
      count++;
    }
    tok = strtok(NULL, delim);
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
  size_t num_fields = count_fields(line, ",");

  Samples samples = {0};
  da_init((&samples), sizeof(Vec));

  do {
    memset(line, 0, sizeof(line));
    fgets(line, sizeof(line), f);
    Vec sample = parse_line(line, num_fields, ",");
    da_append((&samples), sample);
  } while (strlen(line) > 0);

  fclose(f);

  return (Dataset){
      .samples = samples,
      // Last value is the dependent variable
      .dimensions = samples.values[0].size - 1,
      .training_cuttoff = samples.len / 2,
  };
}

Dataset parse_boston_housing_dataset(char *filename) {
  FILE *f = fopen(filename, "rb");
  if (!f) {
    perror("fopen");
    exit(1);
  }

  char line[1024] = {0};
  fgets(line, sizeof(line), f);
  fseek(f, 0, SEEK_SET);
  size_t num_fields = count_fields(line, " ");

  Samples samples = {0};
  da_init((&samples), sizeof(Vec));

  do {
    memset(line, 0, sizeof(line));
    fgets(line, sizeof(line), f);
    Vec sample = parse_line(line, num_fields, " ");
    da_append((&samples), sample);
  } while (strlen(line) > 0);

  fclose(f);
  return (Dataset){
      .samples = samples,
      // Last value is the dependent variable
      .dimensions = samples.values[0].size - 1,
      .training_cuttoff = samples.len / 2,
  };
}

void normalize_dataset(Dataset d) {
  Vec feature_means = new_vec(d.dimensions);
  Vec feature_stddevs = new_vec(d.dimensions);

  size_t n = d.samples.len;
  for (size_t i = 0; i < n; i++) {
    Vec sample = da_at(d.samples, i);
    for (size_t j = 0; j < d.dimensions; j++) {
      VEC_AT(feature_means, j) += (VEC_AT(sample, j) / n);
    }
  }

  for (size_t i = 0; i < n; i++) {
    Vec sample = da_at(d.samples, i);
    for (size_t j = 0; j < d.dimensions; j++) {
      double feature_mean = VEC_AT(feature_means, j);
      VEC_AT(feature_stddevs, j) +=
          pow(VEC_AT(sample, j) - feature_mean, 2) / n;
    }
  }

  for (size_t i = 0; i < d.dimensions; i++) {
    VEC_AT(feature_stddevs, i) = sqrt(VEC_AT(feature_stddevs, i));
  }

  for (size_t i = 0; i < n; i++) {
    Vec sample = da_at(d.samples, i);
    for (size_t j = 0; j < d.dimensions; j++) {
      double feature_mean = VEC_AT(feature_means, j);
      double feature_stddev = VEC_AT(feature_stddevs, j);
      if (feature_stddev != 0.0) {
        VEC_AT(sample, j) = (VEC_AT(sample, j) - feature_mean) / feature_stddev;
      }
    }
  }

  free_vec(feature_means);
  free_vec(feature_stddevs);
}

double predict(Vec w, double b, Vec sample) {
  assert(w.size < sample.size);
  double acc = b;
  for (size_t i = 0; i < w.size; i++) {
    acc += w.values[i] * sample.values[i];
  }

  return acc;
}

void gradient_descent(Dataset d, double eta, Vec w, double *b) {
  Vec d_w = new_vec(w.size);
  double d_b = 0.0;

  size_t n = d.training_cuttoff;
  for (size_t i = 0; i < n; i++) {
    Vec sample = da_at(d.samples, i);
    double expected = EXPECTED(sample);
    double predicted = predict(w, *b, sample);

    double t = (expected - predicted);

    for (size_t j = 0; j < w.size; j++) {
      d_w.values[j] -= sample.values[j] * t;
    }
    d_b -= t;
  }

  for (size_t j = 0; j < w.size; j++) {
    w.values[j] = w.values[j] - 2 * (eta / n) * d_w.values[j];
  }

  *b = *b - 2 * (eta / n) * d_b;

  free_vec(d_w);
}

double validate(Dataset d, Vec w, double b) {
  double mse_acc = 0.0f;
  for (size_t i = d.training_cuttoff; i < d.samples.len; i++) {
    Vec sample = da_at(d.samples, i);
    double expected = EXPECTED(sample);
    double predicted = predict(w, b, sample);
    mse_acc += fabs(expected - predicted) * fabs(expected - predicted);
  }
  return mse_acc / (d.samples.len - d.training_cuttoff);
}

#define EPOCHS 10000000

int main(void) {
  Dataset d = parse_wine_quality_dataset(
      "./linear-regression/data/winequality-red.csv");
  // Dataset d =
  //     parse_boston_housing_dataset("./linear-regression/data/housing.csv");

  normalize_dataset(d);

  Vec w = new_vec(d.dimensions);
  double b = 0.0;
  double prev_loss = INFINITY;
  for (size_t i = 0; i < EPOCHS; i++) {
    gradient_descent(d, 0.0001, w, &b);

    if (i % 10 == 0) {
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
