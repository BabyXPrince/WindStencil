# WindStencil / OpenCFD-SC HIP build
# Layout: include/windstencil (headers), src/{kernels,solver,boundary,io,mpi,runtime,app}

HIPCC     ?= hipcc
MPICXX    ?= mpicxx
ROCM_PATH ?= /opt/rocm
MPI_PATH  ?= /opt/hpc/software/mpi/hpcx/v2.7.4
GPU_ARCH  ?= gfx906

TARGET    ?= opencfd-windstencil

INC       := -I include/windstencil \
             -I $(ROCM_PATH)/include \
             -I $(ROCM_PATH)/include/hip/hcc_detail/cuda

HOST_FLAGS := -std=c99 $(INC) -D__HIP_PLATFORM_AMD__ -D__HIPCC__ -O2
# Kernels/*.c call HIP runtime APIs; compile with hipcc so __HIPCC__ and headers match device objects.
KERNEL_C_FLAGS := -x c -std=c99 $(INC) -I $(MPI_PATH)/include -D__HIP_PLATFORM_AMD__ -D__HIPCC__ -O2
DEV_FLAGS  := -std=c++14 $(INC) -I $(MPI_PATH)/include -mcpu=$(GPU_ARCH) -O2 -D__HIP_PLATFORM_AMD__ -D__HIPCC__

LDFLAGS := -L $(MPI_PATH)/lib -lmpi -lm -lpthread

KERNEL_OBJS  := $(patsubst src/kernels/%.cpp,obj/kernels/%.o,$(wildcard src/kernels/*.cpp)) \
                $(patsubst src/kernels/%.c,obj/kernels/%.o,$(wildcard src/kernels/*.c))
SOLVER_OBJS  := $(patsubst src/solver/%.cpp,obj/solver/%.o,$(wildcard src/solver/*.cpp))
BOUNDARY_OBJS := $(patsubst src/boundary/%.cpp,obj/boundary/%.o,$(wildcard src/boundary/*.cpp)) \
                 $(patsubst src/boundary/%.c,obj/boundary/%.o,$(wildcard src/boundary/*.c))
IO_OBJS      := $(patsubst src/io/%.c,obj/io/%.o,$(wildcard src/io/*.c))
MPI_OBJS     := $(patsubst src/mpi/%.c,obj/mpi/%.o,$(wildcard src/mpi/*.c)) \
                $(patsubst src/mpi/%.cpp,obj/mpi/%.o,$(wildcard src/mpi/*.cpp))
RUNTIME_OBJS := $(patsubst src/runtime/%.c,obj/runtime/%.o,$(wildcard src/runtime/*.c)) \
                $(patsubst src/runtime/%.cpp,obj/runtime/%.o,$(wildcard src/runtime/*.cpp))
APP_SRC      := src/app/opencfd_hip.c
APP_OBJ      := obj/app/opencfd_hip.o

ALL_OBJS := $(KERNEL_OBJS) $(SOLVER_OBJS) $(BOUNDARY_OBJS) $(IO_OBJS) $(MPI_OBJS) $(RUNTIME_OBJS) $(APP_OBJ)

.PHONY: all clean hipify-main help

all: $(TARGET)

help:
	@echo "Targets: all (default), clean, hipify-main"
	@echo "Override: HIPCC, MPICXX, ROCM_PATH, MPI_PATH, GPU_ARCH, TARGET"

$(TARGET): $(ALL_OBJS)
	$(HIPCC) -o $@ $(ALL_OBJS) $(LDFLAGS)

# HIP device / host C++ translation units
obj/kernels/%.o: src/kernels/%.cpp
	@mkdir -p $(dir $@)
	$(HIPCC) $(DEV_FLAGS) -c $< -o $@

obj/solver/%.o: src/solver/%.cpp
	@mkdir -p $(dir $@)
	$(HIPCC) $(DEV_FLAGS) -c $< -o $@

obj/boundary/%.o: src/boundary/%.cpp
	@mkdir -p $(dir $@)
	$(HIPCC) $(DEV_FLAGS) -c $< -o $@

obj/mpi/%.o: src/mpi/%.cpp
	@mkdir -p $(dir $@)
	$(HIPCC) $(DEV_FLAGS) -c $< -o $@

obj/runtime/%.o: src/runtime/%.cpp
	@mkdir -p $(dir $@)
	$(HIPCC) $(DEV_FLAGS) -c $< -o $@

# HIP runtime C sources in kernels/
obj/kernels/%.o: src/kernels/%.c
	@mkdir -p $(dir $@)
	$(HIPCC) $(KERNEL_C_FLAGS) -c $< -o $@

obj/boundary/%.o: src/boundary/%.c
	@mkdir -p $(dir $@)
	$(MPICXX) $(HOST_FLAGS) -c $< -o $@

obj/io/%.o: src/io/%.c
	@mkdir -p $(dir $@)
	$(MPICXX) $(HOST_FLAGS) -c $< -o $@

obj/mpi/%.o: src/mpi/%.c
	@mkdir -p $(dir $@)
	$(MPICXX) $(HOST_FLAGS) -c $< -o $@

obj/runtime/%.o: src/runtime/%.c
	@mkdir -p $(dir $@)
	$(MPICXX) $(HOST_FLAGS) -c $< -o $@

obj/app/%.o: src/app/%.c
	@mkdir -p $(dir $@)
	$(MPICXX) $(HOST_FLAGS) -c $< -o $@

# Regenerate HIP entry from CUDA-era opencfd.c when hipify-perl is available
hipify-main: src/app/opencfd_hip.c

src/app/opencfd_hip.c: src/app/opencfd.c
	hipify-perl $< > $@

clean:
	rm -f $(TARGET) $(ALL_OBJS)
	rm -rf obj
