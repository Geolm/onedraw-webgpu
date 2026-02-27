#include <stdio.h>
#include <string.h>
#include "bin2h.h"

#define STB_TRUETYPE_IMPLEMENTATION
#include "stb_truetype.h"


#define ARENA_NOSTDIO
#define ARENA_IMPLEMENTATION
#include "arena.h"

#define UNUSED_VARIABLE(a) (void)(a)
#define ONEDRAW_MAJOR_VERSION (0)
#define ONEDRAW_MINOR_VERSION (1)


int main(int argc, const char * argv[]) 
{
    UNUSED_VARIABLE(argc);
    UNUSED_VARIABLE(argv);

    Arena arena = {0};

    fprintf(stdout, "onedraw-webgpu %u.%u library builder\n\n", ONEDRAW_MAJOR_VERSION, ONEDRAW_MINOR_VERSION);


    return 0;
}