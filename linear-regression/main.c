#include "../shared/da.h"
#include "../shared/data_structures.h"
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

typedef struct {
  size_t dimensions;
  Samples samples;
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

  printf("File has %ld fields\n", num_fields);

  return (Dataset){
      .samples = samples,
      // Last value is the dependent variable
      .dimensions = samples.values[0].size - 1,
  };
}

int main(void) {
  printf("Hello from linear regression!\n");
  parse_wine_quality_dataset("./linear-regression/data/winequality-red.csv");
  return 0;
}
