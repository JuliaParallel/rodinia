#define CUDA_ERROR_CHECK
#define CUDA
#include "/home/pschen/sslab/src-pschen/omp_offloading/include/cuda_check.h"

#ifndef AT
#ifndef RF
    // 1D function
#define DEEP_COPY1D(NAME, N, TYPE)\
    TYPE *NAME##_d1;\
    {\
        CudaSafeCall(cudaMalloc(&NAME##_d1, N *sizeof(TYPE)));\
        CudaSafeCall(cudaMemcpy(NAME##_d1, NAME, N*sizeof(TYPE), cudaMemcpyHostToDevice));\
    }

#define DEEP_BACK1D(NAME, N, TYPE)\
    CudaSafeCall(cudaMemcpy(NAME, NAME##_d1, N*sizeof(TYPE), cudaMemcpyDeviceToHost));

#define DEEP_FREE1D(NAME)\
    CudaSafeCall(cudaFree(NAME##_d1));

    // 2D function
#define DEEP_COPY2D(NAME, N, M, TYPE) \
    TYPE **NAME##_d1, **NAME##_h1;\
    TYPE *NAME##_d2;\
    { \
        CudaSafeCall(cudaMalloc(&NAME##_d1, N *sizeof(TYPE*)));\
        CudaSafeCall(cudaMalloc(&NAME##_d2, N * M *sizeof(TYPE)));\
        NAME##_h1 = (TYPE**)malloc(N*sizeof(TYPE*)); \
        for (int i = 0 ; i < N; i++) {\
            NAME##_h1[i] = NAME##_d2 +i* M ;\
            CudaSafeCall(cudaMemcpy(NAME##_d2 + i*M, NAME[i], M*sizeof(TYPE), cudaMemcpyHostToDevice));\
        }\
        CudaSafeCall(cudaMemcpy(NAME##_d1, NAME##_h1, N*sizeof(TYPE*), cudaMemcpyHostToDevice));\
    }

#define DEEP_BACK2D(NAME, N, M, TYPE)\
    {\
        for (int i = 0; i < N; i++) {\
            CudaSafeCall(cudaMemcpy(NAME[i], NAME##_d2 + i*M, M * sizeof(TYPE), cudaMemcpyDeviceToHost));\
        }\
    }\

#define DEEP_FREE2D(NAME) \
    {\
        CudaSafeCall(cudaFree(NAME##_d1));\
        CudaSafeCall(cudaFree(NAME##_d2));\
        free(NAME##_h1);\
    }


		// 3D function
#define DEEP_COPY3D(NAME, N, M, K)\
		float ***NAME##_d1, ***NAME##_h1;\
		float *NAME##_d3;\
		float **NAME##_d2;\
		{\
				CudaSafeCall(cudaMalloc(&NAME##_d1, N * sizeof(float**)));\
				CudaSafeCall(cudaMalloc(&NAME##_d2, N * M * sizeof(float*)));\
				CudaSafeCall(cudaMalloc(&NAME##_d3, N * M * K * sizeof(float)));\
				/*Host ptr*/\
				float **NAME##_h2 = (float **)malloc(M * sizeof(float*));\
				NAME##_h1 = (float ***)malloc(N * sizeof(float**));\
				for (int i = 0; i < N; i++) {\
						for (int j = 0; j < M; j++) {\
								CudaSafeCall(cudaMemcpy(NAME##_d3+(i*M+j)*K, NAME[i][j], K * sizeof(float),cudaMemcpyHostToDevice));\
								NAME##_h2[j] = NAME##_d3+ (i*M+j)*K;\
						}\
						CudaSafeCall(cudaMemcpy(NAME##_d2+i*M, NAME##_h2, M * sizeof(float*),cudaMemcpyHostToDevice));\
						NAME##_h1[i] = NAME##_d2+i*M;\
				}\
				CudaSafeCall(cudaMemcpy(NAME##_d1, NAME##_h1, N * sizeof(float**),cudaMemcpyHostToDevice));\
				free(NAME##_h2);\
		}

#define DEEP_BACK3D(NAME, N, M, K)\
		{\
				for (int i = 0; i < N; i ++) {\
						for (int j = 0; j < M; j++) {\
								CudaSafeCall(cudaMemcpy(NAME[i][j], NAME##_d3+(i*M+j)*K, K * sizeof(float), cudaMemcpyDeviceToHost));\
						}\
				}\
		}\


#define DEEP_FREE3D(NAME)\
		CudaSafeCall(cudaFree(NAME##_d3));\
		CudaSafeCall(cudaFree(NAME##_d2));\
		CudaSafeCall(cudaFree(NAME##_d1));\
		free(NAME##_h1);

/******************************Refill RF****************************************************/
#else

#include <vector>
#define REGION_NUM 10

struct ptr_pair {
	char *host;
	char *device;
  int size;
};
struct update_ptr {
  void *ptr;
  void *ptr_addr;
};
std::vector<struct ptr_pair> ptrs;
std::vector<struct update_ptr> update_ptrs;
int threshold = 100;
// FIXME  this should cover OS page unit

// This can be more thin
struct region {
  char *start;
  char *end;
  char *dev_start;
  long bias;
  int used;
};

int inited = 0;

struct region regions[REGION_NUM];

void new_region(void *_addr, size_t size) {
  if (inited == 0) {
    int index = 0;
    while(index < REGION_NUM) {
      regions[index].used = 0;
      index++;
    }
    inited = 1;
  }
  printf("new ptr: %p\n", _addr);
  char *addr;
  addr = (char*)_addr;
  //printf("%p size: %d\n", addr, size);

  int index = 0;
  // insert
  while(index < REGION_NUM) {
    if (regions[index].used == 0) {
      // create new region
      regions[index].used = 1;
      regions[index].start = addr;
      regions[index].end = addr+size;
      //printf("create region %d start: %p\n", index, _addr);
    }

    // judge range
    if (addr < regions[index].start) {
      if (regions[index].start - addr < threshold) {
        regions[index].start = addr;
        //printf("put region %d start ", index);
        break;
      }

    } else {
      if (addr - regions[index].end < threshold) {
        regions[index].end = regions[index].end > (addr+size) ? regions[index].end : addr + size;
        //printf("put region %d end ", index);
        break;
      }
    }

    index++;
    if (index >= REGION_NUM) {
      printf("No enough mem\n");
      exit(-1);
    }
  }
}

void dump_regions() {
  int index =0;
  while (index < REGION_NUM) {
    if (regions[index].used == 0) {
      break;
    }
    struct region *r = &regions[index];
    printf("start: %p, size: %d, dev: %p, bias: %d\n", r->start, r->end - r->start, r->dev_start, r->bias);
    index++;
  }
}

enum REGION_CPY {
  REGION_CPY_D2H,
  REGION_CPY_H2D
};

void ATclean() {
  int index = 0;
  while (index < REGION_NUM) {
    regions[index].used = 0;
    cudaFree(regions[index].dev_start);
    index++;
  }
}

void *LookupAddr(void *addr) {
    int index = 0;
    struct region *cur;
    void *result = nullptr;
    // TODO This can be better
    while(index < REGION_NUM) {
        cur = &regions[index];
        if (cur->used == 0) {
            break;
        }
        if (addr >= cur->start && addr <= cur->end) {
            result = ((char*)addr)+cur->bias;
            break;
        }
        index++;
    }
    if (result == nullptr) {
        printf("Invalid address:%p\n", addr);
    }
    //printf("Translate: %p -> %p\n", addr, result);
    return result;
}

void transfer_regions(enum REGION_CPY type) {
  int index =0;
  while (index < REGION_NUM) {
    if (regions[index].used == 0) {
      break;
    }
    int size = regions[index].end - regions[index].start;
    if (type == REGION_CPY_H2D) {
      char *space;
      CudaSafeCall(cudaMalloc(&space, size));
      regions[index].dev_start = space;
      CudaSafeCall(cudaMemcpy(space, regions[index].start, size, cudaMemcpyHostToDevice));
      regions[index].bias = space-regions[index].start;
    } else if (type == REGION_CPY_D2H) {
      CudaSafeCall(cudaMemcpy(regions[index].start, regions[index].dev_start, size, cudaMemcpyDeviceToHost));
    } else {
      fprintf(stderr, "Unrecognized transfer type\n");
      exit(-1);
    }
    index++;
  }

  puts("Transfer");
// Refilling
	for (auto ptr : ptrs) {
		printf("ptr host_addr %p size: %d\n", ptr.host, ptr.size);
		*(char**)ptr.device = (char*)LookupAddr(ptr.host);
	}
}



void K_LAUNCH(const void* func, int gridDim, int blockDim, unsigned int mask, int n, ...) {
  va_list ap;
	void *args[n];
	void *tmp_addr[n];
	void *arg;
  va_start(ap, n);
  for (int i=0;i<n;i++, mask>>=1)
  {
    arg=va_arg(ap,void*);
		if (mask & 1) {
      void *d_addr = *(void **)arg;
      tmp_addr[i] = LookupAddr(d_addr);
      printf("%p ->", d_addr);
      printf("%p\n", tmp_addr[i]);
      args[i] = &(tmp_addr[i]);
    } else {
      args[i] = arg;
    }
  }
  CudaSafeCall(cudaLaunchKernel (func, dim3(gridDim),dim3( blockDim), args, 0, 0));
	CudaCheckError();
}

void update_ptr(void *ptr, void *ptr_addr) {
  // update new ptr to ptr_addr on device
  ptr = LookupAddr(ptr);
  
}

void add_update_ptr(void *ptr, void *ptr_addr) {

}
    // 1D function
#define DEEP_COPY1D(NAME, N, TYPE)\
    {\
        new_region(NAME, sizeof(TYPE) * N);\
    }

#define DEEP_BACK1D(NAME, N, TYPE)\
    CudaSafeCall(cudaMemcpy(NAME, LookupAddr(NAME), N*sizeof(TYPE), cudaMemcpyDeviceToHost));

#define DEEP_FREE1D(NAME)\
    //CudaSafeCall(cudaFree(NAME##_d1));

    // 2D function
#define DEEP_COPY2D(NAME, N, M, TYPE) \
    { \
        new_region(NAME, sizeof(N) * sizeof(TYPE*));\
        for (int i = 0 ; i < N; i++) {\
            new_region(NAME[i], M* sizeof(TYPE));\
            add_update_ptr(NAME[i], &(NAME[i]));\
        }\
    }
        //push_refill_cb(NAME, N, M

#define DEEP_BACK2D(NAME, N, M, TYPE)\
    {\
        for (int i = 0; i < N; i++) {\
            CudaSafeCall(cudaMemcpy(NAME[i], NAME##_d2 + i*M, M * sizeof(TYPE), cudaMemcpyDeviceToHost));\
        }\
    }\

#define DEEP_FREE2D(NAME) \
    {\
        CudaSafeCall(cudaFree(NAME##_d1));\
        CudaSafeCall(cudaFree(NAME##_d2));\
        free(NAME##_h1);\
    }


		// 3D function
#define DEEP_COPY3D(NAME, N, M, K)\
		float ***NAME##_d1, ***NAME##_h1;\
		float *NAME##_d3;\
		float **NAME##_d2;\
		{\
				CudaSafeCall(cudaMalloc(&NAME##_d1, N * sizeof(float**)));\
				CudaSafeCall(cudaMalloc(&NAME##_d2, N * M * sizeof(float*)));\
				CudaSafeCall(cudaMalloc(&NAME##_d3, N * M * K * sizeof(float)));\
				/*Host ptr*/\
				float **NAME##_h2 = (float **)malloc(M * sizeof(float*));\
				NAME##_h1 = (float ***)malloc(N * sizeof(float**));\
				for (int i = 0; i < N; i++) {\
						for (int j = 0; j < M; j++) {\
								CudaSafeCall(cudaMemcpy(NAME##_d3+(i*M+j)*K, NAME[i][j], K * sizeof(float),cudaMemcpyHostToDevice));\
								NAME##_h2[j] = NAME##_d3+ (i*M+j)*K;\
						}\
						CudaSafeCall(cudaMemcpy(NAME##_d2+i*M, NAME##_h2, M * sizeof(float*),cudaMemcpyHostToDevice));\
						NAME##_h1[i] = NAME##_d2+i*M;\
				}\
				CudaSafeCall(cudaMemcpy(NAME##_d1, NAME##_h1, N * sizeof(float**),cudaMemcpyHostToDevice));\
				free(NAME##_h2);\
		}

#define DEEP_BACK3D(NAME, N, M, K)\
		{\
				for (int i = 0; i < N; i ++) {\
						for (int j = 0; j < M; j++) {\
								CudaSafeCall(cudaMemcpy(NAME[i][j], NAME##_d3+(i*M+j)*K, K * sizeof(float), cudaMemcpyDeviceToHost));\
						}\
				}\
		}\


#define DEEP_FREE3D(NAME)\
		CudaSafeCall(cudaFree(NAME##_d3));\
		CudaSafeCall(cudaFree(NAME##_d2));\
		CudaSafeCall(cudaFree(NAME##_d1));\
		free(NAME##_h1);
#endif
#else
/************************************ AT *****************************************************/
#define REGION_NUM 10

int threshold = 100;
// FIXME  this should cover OS page unit

// This can be more thin
struct region {
  char *start;
  char *end;
  char *dev_start;
  long bias;
  int used;
};

/*
struct host_dev_ptr {
  void *host_addr;
  void *ptr_addr;
};*/

int inited = 0;

__constant__ struct region addr_table[REGION_NUM];

struct region regions[REGION_NUM];

// TODO
/*
void add_array(void *ptr_addr, void *host_addr) {
  struct host_dev_ptr ptr;
  ptr.host_addr = host_addr;
  ptr.ptr_addr = ptr_addr;
  host_dev_ptrs.push_back(ptr);
}*/

void new_region(void *_addr, size_t size) {
  if (inited == 0) {
    int index = 0;
    while(index < REGION_NUM) {
      regions[index].used = 0;
      index++;
    }
    inited = 1;
  }
  char *addr;
  addr = (char*)_addr;
  //printf("%p size: %d\n", addr, size);

  int index = 0;
  // insert
  while(index < REGION_NUM) {
    if (regions[index].used == 0) {
      // create new region
      regions[index].used = 1;
      regions[index].start = addr;
      regions[index].end = addr+size;
      //printf("create region %d start: %p\n", index, _addr);
    }

    // judge range
    if (addr < regions[index].start) {
      if (regions[index].start - addr < threshold) {
        regions[index].start = addr;
        //printf("put region %d start ", index);
        break;
      }

    } else {
      if (addr - regions[index].end < threshold) {
        regions[index].end = regions[index].end > (addr+size) ? regions[index].end : addr + size;
        //printf("put region %d end ", index);
        break;
      }
    }

    index++;
    if (index >= REGION_NUM) {
      printf("No enough mem\n");
      exit(-1);
    }
  }
}

void dump_regions() {
  int index =0;
  while (index < REGION_NUM) {
    if (regions[index].used == 0) {
      break;
    }
    struct region *r = &regions[index];
    printf("start: %p, size: %d, dev: %p, bias: %d\n", r->start, r->end - r->start, r->dev_start, r->bias);
    index++;
  }
}

enum REGION_CPY {
  REGION_CPY_D2H,
  REGION_CPY_H2D
};

void transfer_addr_table() {
  CudaSafeCall(cudaMemcpyToSymbol(addr_table, regions, sizeof(struct region)*REGION_NUM));
}

void ATclean() {
  int index = 0;
  while (index < REGION_NUM) {
    regions[index].used = 0;
    cudaFree(regions[index].dev_start);
    index++;
  }
}

void transfer_regions(enum REGION_CPY type) {
  int index =0;
  while (index < REGION_NUM) {
    if (regions[index].used == 0) {
      break;
    }
    int size = regions[index].end - regions[index].start;
    if (type == REGION_CPY_H2D) {
      char *space;
      CudaSafeCall(cudaMalloc(&space, size));
      regions[index].dev_start = space;
      CudaSafeCall(cudaMemcpy(space, regions[index].start, size, cudaMemcpyHostToDevice));
      regions[index].bias = space-regions[index].start;
      transfer_addr_table();
    } else if (type == REGION_CPY_D2H) {
      CudaSafeCall(cudaMemcpy(regions[index].start, regions[index].dev_start, size, cudaMemcpyDeviceToHost));
    } else {
      fprintf(stderr, "Unrecognized transfer type\n");
      exit(-1);
    }
    index++;
  }
}

    // 1D function
#define DEEP_COPY1D(NAME, N, TYPE)\
    TYPE *NAME##_d1 = NAME;\
    {\
        new_region(NAME, N*sizeof(TYPE));\
    }

#define DEEP_BACK1D(NAME, N, TYPE)

#define DEEP_FREE1D(NAME)

    // 2D function
#define DEEP_COPY2D(NAME, N, M, TYPE) \
    TYPE **NAME##_d1 = NAME;\
    { \
        new_region(NAME, N*sizeof(TYPE *));\
        for (int i = 0 ; i < N; i++) {\
          new_region(NAME[i], M*sizeof(TYPE));\
        }\
    }

#define DEEP_BACK2D(NAME, N, M, TYPE)

#define DEEP_FREE2D(NAME)


		// 3D function
#define DEEP_COPY3D(NAME, N, M, K)\
    float ***NAME##_d1 = NAME;\
		{\
        new_region(NAME, N*sizeof(float**));\
				for (int i = 0; i < N; i++) {\
            new_region(NAME[i], M*sizeof(float*));\
						for (int j = 0; j < M; j++) {\
                new_region(NAME[i][j], K*sizeof(float));\
            }\
        }\
		}

#define DEEP_BACK3D(NAME, N, M, K)

#define DEEP_FREE3D(NAME)

#endif
