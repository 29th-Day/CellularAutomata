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
    _frameDelta = 10000;
    _running = true;
    _pause = false;

    _window = SDL_CreateWindow("CellularAutomata", SDL_WINDOWPOS_CENTERED, SDL_WINDOWPOS_CENTERED, _width * _scale, _height * _scale, SDL_WINDOW_SHOWN);
    _renderer = SDL_CreateRenderer(_window, -1, 0);
    _texture = SDL_CreateTexture(_renderer, SDL_PIXELFORMAT_RGBA8888, SDL_TEXTUREACCESS_STREAMING, _width, _height);
}

Display::~Display()
{
    SDL_DestroyTexture(_texture);
    SDL_DestroyRenderer(_renderer);
    SDL_DestroyWindow(_window);

    SDL_Quit();
}

void Display::draw(float *data)
{
    unsigned int *pixels = nullptr;
    int pitch = 0;

    SDL_LockTexture(_texture, NULL, reinterpret_cast<void **>(&pixels), &pitch);

    for (int i = 0; i < _height * _width; i++)
    {
        unsigned char value = static_cast<unsigned char>(data[i] * 0xFF);
        pixels[i] = (value << 24) | (value << 16) | (value << 8) | 0xFF;
    }

    SDL_UnlockTexture(_texture);
    SDL_RenderCopy(_renderer, _texture, NULL, NULL);
    SDL_RenderPresent(_renderer);
}

bool Display::run()
{
    handleEvents();

    if (_running)
        return true;
    return false;
}

bool Display::nextFrame()
{
    if (_pause)
        return false;

    _frameDelta = SDL_GetTicks() - _frameStart;
    if (_frameDelta >= (unsigned int)(1000 / _fps))
    {
        _frameStart = SDL_GetTicks();
        return true;
    }
    return false;
}

void Display::handleEvents()
{
    SDL_Event event;
    SDL_PollEvent(&event);

    switch (event.type)
    {
    case SDL_QUIT:
        _running = false;
        break;
    // use KEYUP to only activate once and not while holding (hacky)
    case SDL_KEYUP:
        switch (event.key.keysym.sym)
        {
        case SDLK_p:
            _pause = !_pause;
            if (_pause)
                printf("pause\n");
            else
                printf("resume\n");
            break;
        default:
            break;
        }
        break;
    // use KEYDOWN to also use holding
    case SDL_KEYDOWN:
        switch (event.key.keysym.sym)
        {
        case SDLK_UP:
            _fps += 1;
            printf("fps: %u\n", _fps);
            break;
        case SDLK_DOWN:
            _fps -= 1;
            printf("fps: %u\n", _fps);
            break;
        default:
            break;
        }
        break;
    default:
        break;
    }
}
