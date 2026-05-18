#include <stdio.h>
#include <string.h>
#include "bin2h.h"

#define STB_TRUETYPE_IMPLEMENTATION
#include "stb_truetype.h"

#define ARENA_NOSTDIO
#define ARENA_IMPLEMENTATION
#include "arena.h"

#include "../lib/onedraw.h"

// ---------------------------------------------------------------------------------------------------------------------------
#define UNUSED_VARIABLE(a) (void)(a)
#define ONEDRAW_MAJOR_VERSION (0)
#define ONEDRAW_MINOR_VERSION (1)
#define FONT_CHAR_FIRST 33
#define FONT_CHAR_LAST 126
#define FONT_NUM_CHARS 95

// ---------------------------------------------------------------------------------------------------------------------------
void* read_file(const char* filename, size_t* file_size, Arena* arena)
{
    FILE* f = fopen(filename, "rb");
    if (f != NULL)
    {
        fseek(f, 0L, SEEK_END);
        *file_size = ftell(f);
        fseek(f, 0L, SEEK_SET);

        void* buffer = arena_alloc(arena, (*file_size));
        if (fread(buffer, *file_size, 1, f) == 1)
        {
            fclose(f);
            return buffer;
        }
    }
    return NULL;
}

//----------------------------------------------------------------------------------------------------------------------------
char* read_shader(const char* filename, size_t* string_size, Arena* arena)
{
    FILE* f = fopen(filename, "rb");
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
bool export_shaders(Arena* arena)
{
    fprintf(stdout, "=> exporting shaders ");

    size_t common_length;
    char* common_data = read_shader("src/shaders/common.wgsl", &common_length, arena);
    if (common_data == NULL)
    {
        fprintf(stdout, "\ncommon.wgsl not found\n\n");
        return false;
    }

    common_length--;
    
    size_t binning_shader_length;
    char* binning_shader_data = read_shader("src/shaders/binning.wgsl", &binning_shader_length, arena);
    if (binning_shader_data == NULL)
    {
        fprintf(stdout, "\nbinning.wgsl not found\n\n");
        return false;
    }

    size_t combined_length = common_length + binning_shader_length;
    char* combined_shader = arena_alloc(arena, combined_length);
    memcpy(combined_shader, common_data, common_length);
    memcpy(combined_shader + common_length, binning_shader_data, binning_shader_length);

    string2h("lib/binning.h", "binning_shader", combined_shader, combined_length);
    fprintf(stdout, ".");

    size_t rasterizer_shader_length;
    char* rasterizer_shader_data = read_shader("src/shaders/rasterizer.wgsl", &rasterizer_shader_length, arena);
    if (rasterizer_shader_data == NULL)
    {
        fprintf(stdout, "\nshader not found\n\n");
        return false;
    }

    combined_length = common_length + rasterizer_shader_length;
    combined_shader = arena_alloc(arena, combined_length);
    memcpy(combined_shader, common_data, common_length);
    memcpy(combined_shader + common_length, rasterizer_shader_data, rasterizer_shader_length);

    string2h("lib/rasterizer.h", "rasterizer_shader", combined_shader, combined_length);
    fprintf(stdout, ".\n\n");

    return true;
}

// ---------------------------------------------------------------------------------------------------------------------------
bool build_font(float font_height, uint16_t atlas_width, uint16_t atlas_height, Arena *arena)
{
    fprintf(stdout, "=> generating font\n");

    if ((atlas_width%4) != 0 || (atlas_height%4) != 0)
    {
        fprintf(stdout, "font atlas width and height must be a multiple of 4\n");
        return false;
    }

    size_t font_size;
    uint8_t* font_data = read_file("fonts/Satoshi-Medium.otf", &font_size, arena);

    if (font_data == NULL)
        return false;

    fprintf(stdout, "baking %dx%d atlas : ", atlas_width, atlas_height);

    uint8_t* atlas_pixels = arena_alloc(arena, atlas_height * atlas_width);
    if (atlas_pixels == NULL)
        return false;

    stbtt_bakedchar glyphs[FONT_NUM_CHARS];
    int result = stbtt_BakeFontBitmap(font_data, 0, font_height, atlas_pixels, atlas_width, atlas_height,
                                      FONT_CHAR_FIRST, FONT_NUM_CHARS, glyphs);

    if (result == 0)
        return false;
    else if (result < 0)
        fprintf(stdout, "warning only %d chars could fit in the atlas\n", -result);
    else fprintf(stdout, "ok\n");

    // font atlas is uncompressed to target more platforms
    if (!bin2h("lib/default_font_atlas.h", "default_font_atlas", atlas_pixels, atlas_width * atlas_height))
        return false;

    fprintf(stdout, "filling glyphs structure : ");

    od_font* font = arena_alloc(arena, sizeof(od_font));
    if (font == NULL)
        return false;

    memset(font, 0, sizeof(od_font));
    font->first_glyph = FONT_CHAR_FIRST;
    font->num_glyphs = FONT_NUM_CHARS;
    font->texture_width = atlas_width;
    font->texture_height = atlas_height;
    font->font_height = font_height;

    for(uint32_t i=0; i<FONT_NUM_CHARS; ++i)
    {
        font->glyphs[i] = (od_glyph)
        {
            .x0 = glyphs[i].x0,
            .y0 = glyphs[i].y0,
            .x1 = glyphs[i].x1,
            .y1 = glyphs[i].y1,
            .bearing_x = glyphs[i].xoff,
            .bearing_y = glyphs[i].yoff,
            .advance_x = glyphs[i].xadvance
        };
    }

    if (!bin2h("lib/default_font.h", "default_font", font, sizeof(od_font)))
        return false;

    fprintf(stdout, "ok\n\n");
    return true;
}

//----------------------------------------------------------------------------------------------------------------------------
int main(int argc, const char * argv[]) 
{
    UNUSED_VARIABLE(argc);
    UNUSED_VARIABLE(argv);

    Arena arena = {0};

    fprintf(stdout, "onedraw-webgpu %u.%u library builder\n\n", ONEDRAW_MAJOR_VERSION, ONEDRAW_MINOR_VERSION);

    if (!export_shaders(&arena))
        return -1;

    if (!build_font(32.f, 256, 256, &arena))
        return -1;

    arena_free(&arena);

    return 0;
}