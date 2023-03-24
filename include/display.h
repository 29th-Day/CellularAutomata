#pragma once

#include <SDL.h>

class Display
{
    public:
        Display(int height, int width);
        ~Display();

        void draw();

        bool close;

    private:
        void handleEvents();

        SDL_Window *_window;
        SDL_Renderer *_renderer;

        int _height;
        int _width;
};