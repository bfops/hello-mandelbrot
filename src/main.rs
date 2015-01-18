use gl;
use opencl;
use opencl::mem::CLBuffer;
use std::io::timer;
use std::mem;
use std::time::duration::Duration;
use sdl2;
use sdl2::event::Event;
use yaglw::gl_context::{GLContext, GLContextExistence};
use yaglw::shader::Shader;
use yaglw::vertex_buffer::{GLArray, GLBuffer, GLType, VertexAttribData, DrawMode};

static WINDOW_WIDTH: u32 = 800;
static WINDOW_HEIGHT: u32 = 600;

#[repr(C)]
struct RGB {
  pub r: f32,
  pub g: f32,
  pub b: f32,
}

pub fn main() {
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

  let vao = make_picture(&gl, &mut gl_context, &shader);

  while !quit_event() {
    gl_context.clear_buffer();
    vao.draw(&mut gl_context);
    // swap buffers
    window.gl_swap_window();

    timer::sleep(Duration::milliseconds(10));
  }
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
    ((vertex_shader, gl::VERTEX_SHADER)),
    ((fragment_shader, gl::FRAGMENT_SHADER)),
  );

  Shader::new(gl, components.into_iter())
}

fn make_picture<'a>(
  gl: &'a GLContextExistence,
  gl_context: &mut GLContext,
  shader: &Shader<'a>,
) -> GLArray<'a, RGB> {
  let dat = do_opencl();

  let mut vbo = GLBuffer::new(gl, gl_context, dat.len());
  vbo.push(gl_context, dat.as_slice());

  let attribs = [
    VertexAttribData {
      name: "color",
      size: 3,
      unit: GLType::Float,
    },
  ];

  GLArray::new(
    gl,
    gl_context,
    shader,
    &attribs,
    DrawMode::Points,
    vbo,
  )
}

fn do_opencl() -> Vec<RGB> {
  let (device, ctx, queue) = opencl::util::create_compute_context().unwrap();

  let len = WINDOW_WIDTH * WINDOW_HEIGHT;
  let len = len as usize;

  let output_buffer: CLBuffer<RGB> = ctx.create_buffer(len, opencl::cl::CL_MEM_WRITE_ONLY);

  let program = {
    let ker = format!("
      __kernel void color(__global float *output) {{
        int W = {};
        int H = {};
        int i = get_global_id(0);
        float x = i % W;
        float y = i / W;
        i = 3 * i;
        output[i] = 0;
        output[i + 1] = x / W;
        output[i + 2] = y / H;
      }}
    ", WINDOW_WIDTH, WINDOW_HEIGHT);
    ctx.create_program_from_source(ker.as_slice())
  };
  program.build(&device).ok().expect("Couldn't build program.");

  let kernel = program.create_kernel("color");
  // This is sketchy; we "implicitly cast" output_buffer from a CLBuffer<RGB> to a CLBuffer<f32>.
  kernel.set_arg(0, &output_buffer);

  let event = queue.enqueue_async_kernel(&kernel, len, None, ());

  queue.get(&output_buffer, &event)
}

fn make_window() -> sdl2::video::Window {
  sdl2::init(sdl2::INIT_EVERYTHING);

  sdl2::video::gl_set_attribute(sdl2::video::GLAttr::GLContextMajorVersion, 3);
  sdl2::video::gl_set_attribute(sdl2::video::GLAttr::GLContextMinorVersion, 3);
  sdl2::video::gl_set_attribute(
    sdl2::video::GLAttr::GLContextProfileMask,
    sdl2::video::GLProfile::GLCoreProfile as isize
  );

  sdl2::video::Window::new(
    "OpenCL",
    sdl2::video::WindowPos::PosCentered,
    sdl2::video::WindowPos::PosCentered,
    800,
    600,
    sdl2::video::OPENGL,
  ).unwrap()
}

fn quit_event() -> bool {
  loop {
    match sdl2::event::poll_event() {
      Event::None => {
        return false;
      },
      Event::Quit(_) => {
        return true;
      }
      Event::AppTerminating(_) => {
        return true;
      }
      _ => {},
    }
  }
}
