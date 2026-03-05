#include <stdio.h>
#include <string.h>
#include "bin2h.h"

#define STB_TRUETYPE_IMPLEMENTATION
#include "stb_truetype.h"

#define ARENA_NOSTDIO
#define ARENA_IMPLEMENTATION
#include "arena.h"

// ---------------------------------------------------------------------------------------------------------------------------
#define UNUSED_VARIABLE(a) (void)(a)
#define ONEDRAW_MAJOR_VERSION (0)
#define ONEDRAW_MINOR_VERSION (1)

//----------------------------------------------------------------------------------------------------------------------------
char* read_shader(const char* filename, size_t* string_size, Arena* arena)
{
    FILE* f = fopen(filename, "r");
    if (f != NULL)
    {
        fseek(f, 0, SEEK_END);
        long filesize = ftell(f);
        fseek(f, 0, SEEK_SET);

        *string_size = filesize;
        
        char* buffer = (char*) arena_alloc(arena, filesize+1);
        fread(buffer, filesize, 1, f);
        buffer[filesize] = 0;
        fclose(f);
        return buffer;
    }
    return NULL;
}

//----------------------------------------------------------------------------------------------------------------------------
int export_shaders(Arena* arena)
{
    fprintf(stdout, "=> exporting shaders ");
    
    size_t binning_shader_length;
    char* binning_shader_data = read_shader("src/shaders/binning.wgsl", &binning_shader_length, arena);
    if (binning_shader_data == NULL)
    {
        fprintf(stdout, "\nshader not found\n\n");
        return -1;
    }
    string2h("lib/binning.h", "binning_shader", binning_shader_data, binning_shader_length);
    fprintf(stdout, ".");

    size_t rasterizer_shader_length;
    char* rasterizer_shader_data = read_shader("src/shaders/rasterizer.wgsl", &rasterizer_shader_length, arena);
    if (rasterizer_shader_data == NULL)
    {
        fprintf(stdout, "\nshader not found\n\n");
        return -1;
    }
    string2h("lib/rasterizer.h", "rasterizer_shader", rasterizer_shader_data, rasterizer_shader_length);
    fprintf(stdout, ".\n\n");

    return 0;
}

//----------------------------------------------------------------------------------------------------------------------------
int main(int argc, const char * argv[]) 
{
    UNUSED_VARIABLE(argc);
    UNUSED_VARIABLE(argv);

    Arena arena = {0};

    fprintf(stdout, "onedraw-webgpu %u.%u library builder\n\n", ONEDRAW_MAJOR_VERSION, ONEDRAW_MINOR_VERSION);

    if (export_shaders(&arena) != 0)
        return -1;

    arena_free(&arena);

    return 0;
}