use gl;
use mandelbrot::Mandelbrot;
use opencl_context::CL;
use std;
use std::mem;
use sdl2;
use sdl2::event;
use sdl2::event::Event;
use sdl2::keycode::KeyCode;
use sdl2::mouse::Mouse;
use sdl2::video;
use stopwatch::TimerSet;
use yaglw::gl_context::GLContext;
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

  let mut sdl = sdl2::init().everything().unwrap();
  let window = make_window(&sdl);

  let _sdl_gl_context = window.gl_create_context().unwrap();

  // Load the OpenGL function pointers.
  gl::load_with(|s| unsafe {
    mem::transmute(video::gl_get_proc_address(s))
  });

  let mut gl = unsafe {
    GLContext::new()
  };

  match gl.get_error() {
    gl::NO_ERROR => {},
    err => {
      println!("OpenGL error 0x{:x} in setup", err);
      return;
    },
  }

  let shader = make_shader(&gl);
  shader.use_shader(&mut gl);

  let mut vao = make_vao(&mut gl, &shader);
  vao.bind(&mut gl);

  let mut mdlbt = Mandelbrot::new(&cl);
  mdlbt.low_x = -2.0;
  mdlbt.low_y = -2.0;
  mdlbt.width = 4.0;
  mdlbt.height = 4.0;
  mdlbt.max_iter = 128;
  mdlbt.radius = 128.0;

  timers.time("update", || {
    vao.push(&mut gl, mdlbt.render(&cl).as_slice());
  });

  let mut event_pump = sdl.event_pump();

  while process_events(&timers, &mut gl, &mut event_pump, &cl, &mut mdlbt, &mut vao) {
    timers.time("draw", || {
      gl.clear_buffer();
      vao.draw(&mut gl);
      // swap buffers
      window.gl_swap_window();
    });

    std::thread::sleep_ms(10);
  }

  timers.print();
}

fn make_shader<'a, 'b:'a>(
  gl: &'a GLContext,
) -> Shader<'b> {
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

fn make_window(sdl: &sdl2::Sdl) -> video::Window {
  video::gl_attr::set_context_profile(video::GLProfile::Core);
  video::gl_attr::set_context_version(3, 3);

  // Open the window as fullscreen at the current resolution.
  let mut window =
    video::WindowBuilder::new(
      &sdl,
      "Hello, Mandelbrot",
      WINDOW_WIDTH, WINDOW_HEIGHT,
    );

  let window = window.position_centered();
  window.opengl();

  window.build().unwrap()
}

fn make_vao<'a, 'b:'a>(
  gl: &'a mut GLContext,
  shader: &Shader<'b>,
) -> GLArray<'b, RGB> {
  let attribs = [
    VertexAttribData {
      name: "color",
      size: 3,
      unit: GLType::Float,
    },
  ];

  let capacity = WINDOW_WIDTH as usize * WINDOW_HEIGHT as usize;
  let vbo = GLBuffer::new(gl, capacity);

  GLArray::new(
    gl,
    shader,
    &attribs,
    DrawMode::Points,
    vbo,
  )
}

fn process_events<'a>(
  timers: &TimerSet,
  gl: &mut GLContext,
  event_pump: &mut event::EventPump,
  cl: &CL,
  mdlbt: &mut Mandelbrot,
  vao: &mut GLArray<'a, RGB>,
) -> bool {
  loop {
    match event_pump.poll_event() {
      None => {
        return true;
      },
      Some(Event::Quit {..}) => {
        return false;
      },
      Some(Event::AppTerminating {..}) => {
        return false;
      },
      Some(Event::MouseButtonDown {mouse_btn:btn, x, y, ..}) => {
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
      Some(Event::KeyDown {keycode, repeat, ..}) => {
        if !repeat {
          match keycode {
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
