# TPU backend: register "tpu" target kind.

file(GLOB TILE_LANG_TPU_SRCS
  src/tpu/*.cc
)
list(APPEND TILE_LANG_SRCS ${TILE_LANG_TPU_SRCS})
