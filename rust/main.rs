use gl;
use opencl;
use opencl::mem::CLBuffer;
use std::io::File;
use std::io::timer;
use std::mem;
use std::str;
use std::time::duration::Duration;
use sdl2;
use sdl2::event::Event;
use yaglw::gl_context::{GLContext, GLContextExistence};
use yaglw::shader::Shader;
use yaglw::vertex_buffer::{GLArray, GLBuffer, GLType, VertexAttribData, DrawMode};

static WINDOW_WIDTH: u32 = 800;
static WINDOW_HEIGHT: u32 = 600;

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

    in uint sum;

    const int W = {};
    const int H = {};

    out vec4 color;

    void main() {{
      color = vec4(0, float(sum) / (W + H), 0, 1);
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

    in vec4 color;

    void main() {
      gl_FragColor = color;
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
) -> GLArray<'a, u32> {
  let dat = do_opencl();

  let mut vbo = GLBuffer::new(gl, gl_context, dat.len());
  vbo.push(gl_context, dat.as_slice());

  let attribs = [
    VertexAttribData {
      name: "sum",
      size: 1,
      unit: GLType::UInt,
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

fn do_opencl() -> Vec<u32> {
  let (device, ctx, queue) = opencl::util::create_compute_context().unwrap();

  let len = WINDOW_WIDTH * WINDOW_HEIGHT;

  let x_values: Vec<u32> =
    range(0, len).map(|i| i % WINDOW_WIDTH).collect();
  let y_values: Vec<u32> =
    range(0, len).map(|i| i / WINDOW_WIDTH).collect();

  let len = len as usize;

  let x_value_buffer: CLBuffer<u32> = ctx.create_buffer(len, opencl::cl::CL_MEM_READ_ONLY);
  let y_value_buffer: CLBuffer<u32> = ctx.create_buffer(len, opencl::cl::CL_MEM_READ_ONLY);
  let output_buffer: CLBuffer<u32> = ctx.create_buffer(len, opencl::cl::CL_MEM_WRITE_ONLY);

  queue.write(&x_value_buffer, &x_values.as_slice(), ());
  queue.write(&y_value_buffer, &y_values.as_slice(), ());

  let path = Path::new("opencl/add.ocl");
  let ker = File::open(&path).read_to_end().unwrap();
  let program = {
    let ker = str::from_utf8(ker.as_slice()).unwrap();
    ctx.create_program_from_source(ker)
  };
  program.build(&device).ok().expect("Couldn't build program.");

  let kernel = program.create_kernel("vector_add");
  kernel.set_arg(0, &x_value_buffer);
  kernel.set_arg(1, &y_value_buffer);
  kernel.set_arg(2, &output_buffer);

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
