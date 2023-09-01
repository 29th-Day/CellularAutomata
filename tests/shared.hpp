#pragma once

#include <doctest/doctest.h>

#define types int, char, unsigned int, float

#define CHECK_ARRAY(a, b, l) do { for (int i = 0; i < l; i++) { CHECK_MESSAGE(a[i] == doctest::Approx(b[i]), "index ", i); } } while(false)
