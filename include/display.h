#pragma once

#include <SDL.h>

class Display
{
    public:
        Display(int height, int width, int scale = 1, int fps = 10);
        ~Display();

        void draw(float *data);
        bool run();
        bool nextFrame();
    private:
        void handleEvents();
        void wait();

        SDL_Window *_window;
        SDL_Renderer *_renderer;
        SDL_Texture *_texture;

        int _fps;
        int _height;
        int _width;
        int _scale;

        bool _running;
        unsigned long _frameStart;
        unsigned long _frameDelta;
};