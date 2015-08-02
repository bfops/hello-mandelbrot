use main::{WINDOW_WIDTH, WINDOW_HEIGHT, RGB};
use opencl;
use opencl::hl::{Kernel};
use opencl::mem::CLBuffer;
use opencl_context::CL;
use std::borrow::Borrow;

pub struct Mandelbrot {
  pub low_x: f64,
  pub low_y: f64,
  pub width: f64,
  pub height: f64,
  pub max_iter: u32,
  pub radius: f64,

  output_buffer: CLBuffer<RGB>,
  len: usize,
  kernel: Kernel,
}

impl Mandelbrot {
  pub fn new(cl: &CL) -> Mandelbrot {
    let len = WINDOW_WIDTH as usize * WINDOW_HEIGHT as usize;

    let program = {
      let ker = format!("
          __kernel void color(
            const double low_x,
            const double low_y,
            const double width,
            const double height,
            const int max_iter,
            const double radius,
            __global float * output)
          {{
            int W = {};
            int H = {};

            int i = get_global_id(0);

            double c_x = i % W;
            double c_y = i / W;
            c_x = c_x * width / W + low_x;
            c_y = c_y * height / H + low_y;

            double x = 0;
            double y = 0;
            int it;
            for (it = 0; it < max_iter; ++it)
            {{
              double x2 = x * x;
              double y2 = y * y;
              if (x2 + y2 > radius * radius)
                break;
              // Ordering is important here.
              y = 2*x*y + c_y;
              x = x2 - y2 + c_x;
            }}

            i = i * 3;

            if (it < max_iter) {{
              float progress = (float)it / (float)max_iter;
              output[i] = progress;
              output[i + 1] = 1 - 2 * fabs(0.5 - progress);
              output[i + 2] = 0.5 * (1 - progress);
            }} else {{
              output[i] = 0;
              output[i + 1] = 0;
              output[i + 2] = 0;
            }}
          }}
        ", WINDOW_WIDTH, WINDOW_HEIGHT);
      cl.context.create_program_from_source(ker.borrow())
    };
    program.build(&cl.device).unwrap();

    let kernel = program.create_kernel("color");

    Mandelbrot {
      low_x: 0.0,
      low_y: 0.0,
      width: 0.0,
      height: 0.0,
      max_iter: 0,
      radius: 0.0,

      output_buffer: cl.context.create_buffer(len, opencl::cl::CL_MEM_WRITE_ONLY),
      kernel: kernel,
      len: len,
    }
  }

  pub fn render(&self, cl: &CL) -> Vec<RGB> {
    self.kernel.set_arg(0, &self.low_x);
    self.kernel.set_arg(1, &self.low_y);
    self.kernel.set_arg(2, &self.width);
    self.kernel.set_arg(3, &self.height);
    self.kernel.set_arg(4, &self.max_iter);
    self.kernel.set_arg(5, &self.radius);

    // This is sketchy; we "implicitly cast" output_buffer from a CLBuffer<RGB> to a CLBuffer<f32>.
    self.kernel.set_arg(6, &self.output_buffer);

    let event = cl.queue.enqueue_async_kernel(&self.kernel, self.len, None, ());
    cl.queue.get(&self.output_buffer, &event)
  }
}
