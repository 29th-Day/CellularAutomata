#pragma once

#include <SDL.h>
#include <thread>

class Display
{
    public:
        Display(int height, int width);
        ~Display();

        void start();
        void stop();

        void draw(float **data);

        volatile bool running;

    private:
        void run();

        void handleEvents();

        SDL_Window *_window;
        SDL_Renderer *_renderer;
        SDL_Texture *_texture;

        int _height;
        int _width;
};