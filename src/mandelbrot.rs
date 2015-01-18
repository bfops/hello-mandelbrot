use main::{WINDOW_WIDTH, WINDOW_HEIGHT, RGB};
use opencl;
use opencl::mem::CLBuffer;

pub struct Mandelbrot {
  pub low_x: f32,
  pub low_y: f32,
  pub width: f32,
  pub height: f32,
  pub max_iter: u32,
}

impl Mandelbrot {
  pub fn render(&self) -> Vec<RGB> {
    let (device, ctx, queue) = opencl::util::create_compute_context().unwrap();

    let len = WINDOW_WIDTH as usize * WINDOW_HEIGHT as usize;

    let output_buffer: CLBuffer<RGB> = ctx.create_buffer(len, opencl::cl::CL_MEM_WRITE_ONLY);

    let program = {
      let ker = format!("
          __kernel void color(
            const float low_x,
            const float low_y,
            const float width,
            const float height,
            const int max_iter,
            __global float * output)
          {{
            int W = {};
            int H = {};

            float R = 100;

            int i = get_global_id(0);

            float c_x = i % W;
            float c_y = i / W;
            c_x = (c_x / W) * width + low_x;
            c_y = (c_y / H) * height + low_y;

            float x = 0;
            float y = 0;
            int it;
            for (it = 0; it < max_iter; ++it)
            {{
              float x2 = x * x;
              float y2 = y * y;
              if (x2 + y2 > R * R)
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
      ctx.create_program_from_source(ker.as_slice())
    };
    program.build(&device).unwrap();

    let kernel = program.create_kernel("color");
    kernel.set_arg(0, &self.low_x);
    kernel.set_arg(1, &self.low_y);
    kernel.set_arg(2, &self.width);
    kernel.set_arg(3, &self.height);
    kernel.set_arg(4, &self.max_iter);

    // This is sketchy; we "implicitly cast" output_buffer from a CLBuffer<RGB> to a CLBuffer<f32>.
    kernel.set_arg(5, &output_buffer);

    let event = queue.enqueue_async_kernel(&kernel, len, None, ());

    queue.get(&output_buffer, &event)
  }
}
