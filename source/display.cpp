#include "display.h"

#include <SDL.h>

Display::Display(int height, int width, int scale, int fps)
{
    SDL_Init(SDL_INIT_VIDEO | SDL_INIT_EVENTS | SDL_INIT_TIMER);

    _height = height;
    _width = width;
    _scale = scale;
    _fps = fps;

    _frameStart = 0;
    _frameDelta = 0;

    running = true;

    char title[100];
    sprintf_s(title, "CellularAutomata (%ux%u)", _width, _height);

    _window = SDL_CreateWindow(title, SDL_WINDOWPOS_CENTERED, SDL_WINDOWPOS_CENTERED, _width * _scale, _height * _scale, SDL_WINDOW_SHOWN);
    _renderer = SDL_CreateRenderer(_window, -1, 0);
    _texture = SDL_CreateTexture(_renderer, SDL_PIXELFORMAT_RGB332, SDL_TEXTUREACCESS_STREAMING, _width, _height);
}

Display::~Display()
{
    SDL_DestroyTexture(_texture);
    SDL_DestroyRenderer(_renderer);
    SDL_DestroyWindow(_window);
}

void Display::draw(float *data)
{
    unsigned char *pixels = NULL;
    int pitch = NULL;

    SDL_LockTexture(_texture, NULL, (void**)&pixels, &pitch);

    for (int i = 0; i < _height * _width; i++)
    {
        pixels[i] = (unsigned char)(data[i] * 0xFF);
    }

    SDL_UnlockTexture(_texture);
    SDL_RenderCopy(_renderer, _texture, NULL, NULL);
    SDL_RenderPresent(_renderer);
}

bool Display::run()
{
    handleEvents();

    _frameDelta = SDL_GetTicks() - _frameStart;
    SDL_Delay(__max(0, (1000 / _fps) - _frameDelta));
    _frameStart = SDL_GetTicks();

    if (running)
        return true;
    
    return false;
}

void Display::handleEvents()
{
    SDL_Event event;
    SDL_PollEvent(&event);

    switch (event.type)
    {
    case SDL_QUIT:
        running = false;
        break;
    
    default:
        break;
    }
}

void Display::wait()
{
    SDL_Delay(30);
}