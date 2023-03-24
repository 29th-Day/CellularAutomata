#include "display.h"

#include "SDL.h"

Display::Display(int height, int width)
{
    SDL_Init(SDL_INIT_VIDEO | SDL_INIT_EVENTS | SDL_INIT_TIMER);

    _window = SDL_CreateWindow("Cellular Automata", 100, 100, width, height, SDL_WINDOW_SHOWN);
    _renderer = SDL_CreateRenderer(_window, -1, 0);

    close = false;
}

Display::~Display()
{
    
}

void Display::draw()
{
    handleEvents();
}

void Display::handleEvents()
{
    SDL_Event event;
    SDL_PollEvent(&event);

    switch (event.type)
    {
    case SDL_QUIT:
        close = true;
        break;
    
    default:
        break;
    }
}