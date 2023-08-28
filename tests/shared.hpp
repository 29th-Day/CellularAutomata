#pragma once

#define CHECK_ARRAY(a, b, l) do { for (int i = 0; i < l; i++) { CHECK_MESSAGE(a[i] == b[i], "index ", i); } } while(false)
