use opencl;
use opencl::mem::CLBuffer;
use std::io::File;
use std::str;

static WINDOW_WIDTH: uint = 800;
static WINDOW_HEIGHT: uint = 600;

pub fn main() {
  let path = Path::new("opencl/add.ocl");
  let ker = File::open(&path).read_to_end().unwrap();
  let ker = str::from_utf8(ker.as_slice()).unwrap();

  let (device, ctx, queue) = opencl::util::create_compute_context().unwrap();

  let len = WINDOW_WIDTH * WINDOW_HEIGHT;

  let x_values: Vec<uint> =
    range(0, len).map(|i| i % WINDOW_WIDTH).collect();
  let y_values: Vec<uint> =
    range(0, len).map(|i| i / WINDOW_WIDTH).collect();

  let x_value_buffer: CLBuffer<uint> = ctx.create_buffer(len, opencl::CL::CL_MEM_READ_ONLY);
  let y_value_buffer: CLBuffer<uint> = ctx.create_buffer(len, opencl::CL::CL_MEM_READ_ONLY);
  let output_buffer: CLBuffer<uint> = ctx.create_buffer(len, opencl::CL::CL_MEM_WRITE_ONLY);

  queue.write(&x_value_buffer, &x_values.as_slice(), ());
  queue.write(&y_value_buffer, &y_values.as_slice(), ());

  let program = ctx.create_program_from_source(ker);
  program.build(&device).ok().expect("Couldn't build program.");

  let kernel = program.create_kernel("vector_add");
  kernel.set_arg(0, &x_value_buffer);
  kernel.set_arg(1, &y_value_buffer);
  kernel.set_arg(2, &output_buffer);

  let event = queue.enqueue_async_kernel(&kernel, len, None, ());

  let vec_c: Vec<uint> = queue.get(&output_buffer, &event);
  println!("results: {}", vec_c);
}