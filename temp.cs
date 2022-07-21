안녕하세요.뉴로클 기술지원팀 입니다.
보내주신 내용을 바탕으로 코드를 분석해보았습니다.
그 결과 2가지 오류를 확인하였습니다.

결과값을 get_at 할 때에, model에 설정되어있는 pred.prob idx를 가져오는 것이 아니라, 
0, 1로 static한 값을 넣어주고 있었습니다. 

모델을 얻어온 후, 값을 확인할 때에는 아래와 같은 코드를 추가하신 후, 
이후에 get_at 해올 때 idx값을 넣어주어야합니다. 



int pred_idx = -1, prob_idx = -1;
int num_outputs = model.get_num_outputs(); // my model 에서는 이게 0으로 나오네요

for (int i = 0; i < num_outputs; i++)
{
  if (model.get_output_flag(i) == nrt.Model.MODELIO_OUT_PRED)
  {
    pred_idx = i;
    Console.Write("output pred idx" + i);
  }

  if (model.get_output_flag(i) == nrt.Model.MODELIO_OUT_PROB)
  {
    prob_idx = i;
    Console.Write("output prob idx" + i);
  }

  nrt.Shape shp = model.get_output_shape(i);
  Console.Write("output " + i + " " + model.get_output_name(i) + " [");

  for (int j = 0; j < shp.num_dim; j++)
  {
    Console.Write(shp.get_axis(j) + " ");
  }
  Console.WriteLine("] DType: " + nrt.nrt.dtype_to_str(model.get_output_dtype(i)));
}



// 이 후의 결과값을 얻어올 때, 이 코드는 patchmode의 코드이고, 
// patchmode가 아닐때의 get_at에서도 idx를 static하게 가져가선 안됨.

status = nrt.nrt.merge_patches_to_orginal_shape(outputs.get_at(prob_idx), patch_info, prob_map);
if (status != nrt.Status.STATUS_SUCCESS)
{
  Console.WriteLine("merge_patches_to_orginal_shape failed.  : " + nrt.nrt.get_last_error_msg());
  return;
}

status = nrt.nrt.merge_patches_to_orginal_shape(outputs.get_at(pred_idx), patch_info, merged_output);
if (status != nrt.Status.STATUS_SUCCESS)
{
  Console.WriteLine("merge_patches_to_orginal_shape failed.  : " + nrt.nrt.get_last_error_msg());
  return;
}





2.사용하신 이미지는 1채널 이미지인데, 이에대한 반영이 되어있지 않았습니다. 

// 아래의 input 크기는 원본 이미지 크기입니다. 채널의 경우 저희는 아직 3채널만 지원합니다.
int input_h = 2048;
int input_w = 2048;
/*
In the current version, all input images are processed as 3 channels. Grayscale images must be converted to 3 channels.
현재 버전에서는 모든 입력 영상이 3채널로 처리됩니다. 그레이스케일 영상은 3개의 채널로 변환되어야 합니다.
*/
int input_c = 3;
int input_image_byte_size = input_h * input_w * input_c;
//
//이 방법 대신 아래의 byte_buff를 읽어오는 방법을 사용합니다.
//이대로 사용하셨기 때문에, image_paths로 읽어온 이미지는 1채널이지만, NDBuffer의 크기는 3채널로잡혀있어 복사에 대한 오류가 생긴 것으로 파악됩니다.
//nrt.NDBuffer images = nrt.NDBuffer.load_images(new nrt.Shape(input_h, input_w, input_c), image_paths, resize_method);

MemoryStream ms = new MemoryStream();
Image img = Image.FromFile(image_paths);
img.Save(ms, ImageFormat.Bmp);
byte[] byte_buff = ms.ToArray();

nrt.NDBuffer image_buff = new nrt.NDBuffer(new nrt.Shape(current_batch_size, input_h, input_w, input_c), input_dtype);
for (int j = 0; j < current_batch_size; j++)
{
    image_buff.copy_from_buffer_uint8(j, byte_buff, (ulong)byte_buff.Length); // Copy one image to each batch location
}

위 코드는 batch_size가 1일 때를 가정한 경우로, 
batch_size를 좀더 크게 가져가고 싶으신 경우에는 해당 부분을 수정하시면 됩니다.
위와 같이 수정했을 때에 잘 나오는 것을 확인하였으며, 
추가적으로 질문이 있으실 경우 연락부탁드립니다.