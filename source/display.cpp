#include "display.h"

#include <SDL.h>
#include <thread>

Display::Display(int height, int width)
{
    SDL_Init(SDL_INIT_VIDEO | SDL_INIT_EVENTS | SDL_INIT_TIMER);

    _height = height;
    _width = width;
}

Display::~Display()
{
    stop();
}

void Display::start()
{
    running = true;

    _window = SDL_CreateWindow("Cellular Automata", 100, 100, _width, _height, SDL_WINDOW_SHOWN);
    _renderer = SDL_CreateRenderer(_window, -1, 0);
    _texture = SDL_CreateTexture(_renderer, SDL_PIXELFORMAT_RGB332, SDL_TEXTUREACCESS_STREAMING, _width, _height);


    std::thread _thread(&Display::run, this);
    _thread.detach();
}

void Display::stop()
{
    running = false;

    SDL_DestroyTexture(_texture);
    SDL_DestroyRenderer(_renderer);
    SDL_DestroyWindow(_window);
}

void Display::run()
{
    while(running)
    {
        handleEvents();
    }
}

void Display::draw(float **data)
{
    int *buffer = new int[_height * _width]();

    for (int i = 0; i < _height * _width; i++)
    {
        // buffer[i] = (int)(*data[i] * 0xFF);
        buffer[i] = 0b11100000;
    }

    int pitch = sizeof(int) * _width;
    SDL_LockTexture(_texture, NULL, (void**)&buffer, &pitch);

    SDL_UnlockTexture(_texture);

    SDL_RenderCopy(_renderer, _texture, NULL, NULL);

    // handleEvents();
}

void Display::handleEvents()
{
    SDL_Event event;
    SDL_PollEvent(&event);

    switch (event.type)
    {
    case SDL_QUIT:
        stop();
        break;
    
    default:
        break;
    }
}