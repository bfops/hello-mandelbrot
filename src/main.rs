use gl;
use mandelbrot::Mandelbrot;
use opencl_context::CL;
use std::io::timer;
use std::mem;
use std::time::duration::Duration;
use sdl2;
use sdl2::event::Event;
use sdl2::keycode::KeyCode;
use sdl2::mouse::Mouse;
use stopwatch::TimerSet;
use yaglw::gl_context::{GLContext, GLContextExistence};
use yaglw::shader::Shader;
use yaglw::vertex_buffer::{GLArray, GLBuffer, GLType, VertexAttribData, DrawMode};

pub const WINDOW_WIDTH: u32 = 800;
pub const WINDOW_HEIGHT: u32 = 800;

#[repr(C)]
pub struct RGB {
  pub r: f32,
  pub g: f32,
  pub b: f32,
}

pub fn main() {
  let timers = TimerSet::new();

  let cl = unsafe {
    CL::new()
  };

  let window = make_window();

  let _sdl_gl_context = window.gl_create_context().unwrap();

  // Load the OpenGL function pointers.
  gl::load_with(|s| unsafe {
    mem::transmute(sdl2::video::gl_get_proc_address(s))
  });

  let (gl, mut gl_context) = unsafe {
    GLContext::new()
  };

  match gl_context.get_error() {
    gl::NO_ERROR => {},
    err => {
      println!("OpenGL error 0x{:x} in setup", err);
      return;
    },
  }

  let shader = make_shader(&gl);
  shader.use_shader(&mut gl_context);

  let mut vao = make_vao(&gl, &mut gl_context, &shader);
  vao.bind(&mut gl_context);

  let mut mdlbt =
    Mandelbrot {
      low_x: -2.0,
      low_y: -2.0,
      width: 4.0,
      height: 4.0,
      max_iter: 128,
      radius: 128.0,
      ..Mandelbrot::new(&cl)
    };

  timers.time("update", || {
    vao.push(&mut gl_context, mdlbt.render(&cl).as_slice());
  });

  while process_events(&timers, &mut gl_context, &cl, &mut mdlbt, &mut vao) {
    timers.time("draw", || {
      gl_context.clear_buffer();
      vao.draw(&mut gl_context);
      // swap buffers
      window.gl_swap_window();
    });

    timer::sleep(Duration::milliseconds(10));
  }

  timers.print();
}

fn make_shader<'a>(
  gl: &'a GLContextExistence,
) -> Shader<'a> {
  let vertex_shader: String = format!("
    #version 330 core

    const int W = {};
    const int H = {};

    in vec3 color;
    out vec4 v_color;

    void main() {{
      v_color = vec4(color, 1);
      gl_Position =
        vec4(
          float(gl_VertexID % W) / W * 2 - 1,
          float(gl_VertexID / W) / H * 2 - 1,
          0, 1
        );
    }}
  ", WINDOW_WIDTH, WINDOW_HEIGHT);

  let fragment_shader: String = "
    #version 330 core

    in vec4 v_color;

    void main() {
      gl_FragColor = v_color;
    }
  ".to_string();

  let components = vec!(
    (gl::VERTEX_SHADER, vertex_shader),
    (gl::FRAGMENT_SHADER, fragment_shader),
  );

  Shader::new(gl, components.into_iter())
}

fn make_window() -> sdl2::video::Window {
  sdl2::init(sdl2::INIT_EVERYTHING);

  sdl2::video::gl_set_attribute(sdl2::video::GLAttr::GLContextMajorVersion, 3);
  sdl2::video::gl_set_attribute(sdl2::video::GLAttr::GLContextMinorVersion, 3);
  sdl2::video::gl_set_attribute(
    sdl2::video::GLAttr::GLContextProfileMask,
    sdl2::video::GLProfile::GLCoreProfile as isize
  );

  let window = sdl2::video::Window::new(
    "OpenCL",
    sdl2::video::WindowPos::PosCentered,
    sdl2::video::WindowPos::PosCentered,
    WINDOW_WIDTH as isize,
    WINDOW_HEIGHT as isize,
    sdl2::video::OPENGL,
  ).unwrap();

  window
}

fn make_vao<'a>(
  gl: &'a GLContextExistence,
  gl_context: &mut GLContext,
  shader: &Shader<'a>,
) -> GLArray<'a, RGB> {
  let attribs = [
    VertexAttribData {
      name: "color",
      size: 3,
      unit: GLType::Float,
    },
  ];

  let capacity = WINDOW_WIDTH as usize * WINDOW_HEIGHT as usize;
  let vbo = GLBuffer::new(gl, gl_context, capacity);

  GLArray::new(
    gl,
    gl_context,
    shader,
    &attribs,
    DrawMode::Points,
    vbo,
  )
}

fn process_events<'a>(
  timers: &TimerSet,
  gl: &mut GLContext,
  cl: &CL,
  mdlbt: &mut Mandelbrot,
  vao: &mut GLArray<'a, RGB>,
) -> bool {
  loop {
    match sdl2::event::poll_event() {
      Event::None => {
        return true;
      },
      Event::Quit(_) => {
        return false;
      },
      Event::AppTerminating(_) => {
        return false;
      },
      Event::MouseButtonDown(_, _, _, btn, x, y) => {
        let y = WINDOW_HEIGHT as i32 - y;
        match btn {
          Mouse::Left => {
            let ww = WINDOW_WIDTH as f64;
            let wh = WINDOW_HEIGHT as f64;
            mdlbt.low_x += (x as f64) * mdlbt.width / ww;
            mdlbt.low_y += (y as f64) * mdlbt.height / wh;
            mdlbt.width /= 4.0;
            mdlbt.height /= 4.0;
            mdlbt.low_x -= mdlbt.width / 2.0;
            mdlbt.low_y -= mdlbt.height / 2.0;

            timers.time("update", || {
              vao.buffer.update(gl, 0, mdlbt.render(cl).as_slice());
            });
          },
          Mouse::Right => {
            let ww = WINDOW_WIDTH as f64;
            let wh = WINDOW_HEIGHT as f64;
            mdlbt.low_x += mdlbt.width / 2.0;
            mdlbt.low_y += mdlbt.height / 2.0;
            mdlbt.width *= 4.0;
            mdlbt.height *= 4.0;
            mdlbt.low_x -= (x as f64) * mdlbt.width / ww;
            mdlbt.low_y -= (y as f64) * mdlbt.height / wh;

            timers.time("update", || {
              vao.buffer.update(gl, 0, mdlbt.render(cl).as_slice());
            });
          }
          _ => {},
        };
      },
      Event::KeyDown(_, _, key, _, _, repeat) => {
        if !repeat {
          match key {
            KeyCode::Up => {
              mdlbt.max_iter *= 2;

              timers.time("update", || {
                vao.buffer.update(gl, 0, mdlbt.render(cl).as_slice());
              });
            },
            KeyCode::Down => {
              mdlbt.max_iter /= 2;

              timers.time("update", || {
                vao.buffer.update(gl, 0, mdlbt.render(cl).as_slice());
              });
            },
            KeyCode::Right => {
              mdlbt.radius *= 2.0;

              timers.time("update", || {
                vao.buffer.update(gl, 0, mdlbt.render(cl).as_slice());
              });
            },
            KeyCode::Left => {
              mdlbt.radius /= 2.0;

              timers.time("update", || {
                vao.buffer.update(gl, 0, mdlbt.render(cl).as_slice());
              });
            },
            _ => {},
          }
        }
      },
      _ => {},
    }
  }
}
