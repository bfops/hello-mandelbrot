extern crate gl;
#[macro_use]
extern crate log;
extern crate opencl;
extern crate sdl2;
extern crate time;
extern crate yaglw;

mod main;
mod mandelbrot;
mod stopwatch;

pub fn main() {
  main::main();
}
