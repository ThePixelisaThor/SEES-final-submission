/*
    Main code for the course of Software engineering and embedded systems.
    It implements a custom gesture recognition ML model, in dual core
*/

#include "main_sees.hpp"

static int16_t *input_ptr = NULL; // pointer to the inputs of the ML model
static int16_t *output_ptr = NULL;// pointer to the outputs of the ML model
static int8_t exponent; // quantization exponent of the model

// "_binary_model_espdl_start" is composed of three parts: the prefix "binary", the filename "model_espdl", and the suffix "_start".
extern const uint8_t model_espdl[] asm("_binary_handrecognition_model4_espdl_start");
extern const uint8_t model_espdl_end[] asm("_binary_handrecognition_model4_espdl_end");

static dl::Model *model = NULL;

// Different gestures, mapping between the index and the gesture
const char *classes[] = {
    "01_fist", "02_up", "03_down", "04_left",
    "05_right", "06_forward", "07_backward", "08_nothing"};

static const char *TAG = "main_sees"; // debug tag for ESP console

// Flags to allow for an easy dual core implementation
volatile bool is_processing = false; // is the model currently processing a frame (ie it shouldn't write a new frame)
volatile bool is_new_image = false; // has a new image arrived (ie should the model be run)

static jpeg_enc_config_t jpeg_enc_cfg; // config for the jpeg encoding
static jpeg_enc_handle_t jpeg_enc;

uint8_t ml_frame_rgb[3 * 45 * 45]; // buffer for the cropped rgb frame, rgb888
uint8_t ml_frame_jpeg[3 * 45 * 45]; // buffer for the encoded jpeg frame (that will be sent by wifi)
int ml_frame_jpeg_len; // length of the encoded jpeg frame (not fixed length unlike rgb888)

uint8_t max_index_byte = 7; // current result of the ML model, initialized at nothing by default

uint8_t get_result() {
    return max_index_byte;
}

uint8_t *get_rgb_frame()
{
    return &ml_frame_rgb[0];
}

uint8_t *get_jpeg_frame()
{
    return &ml_frame_jpeg[0];
}

size_t get_jpeg_len()
{
    return (size_t)ml_frame_jpeg_len;
}

void initialize_model_jpeg_spi()
{
    ESP_LOGW(TAG, "Creating model");
    // Loads model from flash
    model = new dl::Model((const char *)model_espdl, fbs::MODEL_LOCATION_IN_FLASH_RODATA);

    // Initialization of the inputs and outputs
    std::map<std::string, dl::TensorBase *> model_inputs = model->get_inputs();
    dl::TensorBase *model_input = model_inputs.begin()->second;
    std::map<std::string, dl::TensorBase *> model_outputs = model->get_outputs();
    dl::TensorBase *model_output = model_outputs.begin()->second;

    // Setting the pointers
    input_ptr = (int16_t *)model_input->data;
    output_ptr = (int16_t *)model_output->data;
    exponent = model_input->exponent;

    // Setting the ML frame to 0
    for (int i = 0; i < 3 * 45 * 45; i++)
    {
        ml_frame_rgb[i] = 0;
    }

    ESP_LOGW(TAG, "Model was created");
    ESP_ERROR_CHECK(model->test()); // this uses a saved test vector in the model to validate that it is loaded with the correct quantization
    ESP_LOGW(TAG, "Model was tested");

    ESP_LOGW(TAG, "Initializing JPEG"); // this initializes the jpeg encoder

    jpeg_enc_cfg = DEFAULT_JPEG_ENC_CONFIG();
    jpeg_enc_cfg.width = 45;
    jpeg_enc_cfg.height = 45;
    jpeg_enc_cfg.src_type = JPEG_PIXEL_FORMAT_RGB888;
    jpeg_enc_cfg.subsampling = JPEG_SUBSAMPLE_420;
    jpeg_enc_cfg.quality = 100;
    jpeg_enc_cfg.rotate = JPEG_ROTATE_0D;
    jpeg_enc_cfg.task_enable = false;
    jpeg_enc_cfg.hfm_task_priority = 13;
    jpeg_enc_cfg.hfm_task_core = 1;

    jpeg_error_t ret = JPEG_ERR_OK;

    jpeg_enc = NULL;

    // open
    ret = jpeg_enc_open(&jpeg_enc_cfg, &jpeg_enc);
    if (ret != JPEG_ERR_OK)
    {
        ESP_LOGE(TAG, "Error creating encoder");
        return;
    }
}

// Debug function, not used anymore
static void print_red_channel_ascii(uint8_t *rgb_image, int width, int height)
{
    // Characters to represent different intensity levels
    const char *ascii_table = " .:-=+*%@#";

    // Iterate through each pixel in the image
    for (int y = 0; y < height; y++)
    {
        for (int x = 0; x < width; x++)
        {
            // Get the red intensity (8 bits from the RGB888 format)
            uint8_t red = rgb_image[3 * (y * width + x)]; // Red is the first byte in RGB888

            // Map red intensity to a range of ASCII characters
            int index = (red * 10) / 256; // Scale to [0, 9]
            printf("%c", ascii_table[index]);
            printf("%c", ascii_table[index]);
        }
        printf("\n");
    }
}

// Debug function, not used anymore
static void print_red_channel_ascii2(uint8_t *rgb_image, int width, int height)
{
    // Iterate through each pixel in the image
    for (int y = 0; y < height; y++)
    {
        for (int x = 0; x < width; x++)
        {
            // Get the red intensity (8 bits from the RGB888 format)
            uint8_t red = rgb_image[3 * (y * width + x)]; // Red is the first byte in RGB888
            printf("%d ", red);
        }
        printf("\n");
    }
}

// Function called when a frame is received by the webcam
void push_frame(uvc_frame_t *frame)
{

    uint8_t *decoded;
    // The webcam resolutoin has to be set to 320x180 for the USB OTG to work
    // The frame will be rescaled by 1/4
    int decoded_outsize = 3 * 320 / 4 * 180 / 4; 
    decoded = (uint8_t *)malloc(decoded_outsize);

    // This is the config to decode a frame
    esp_jpeg_image_cfg_t jpeg_cfg = {
        .indata = (uint8_t *)frame->data,
        .indata_size = frame->data_bytes,
        .outbuf = decoded,
        .outbuf_size = (uint32_t)decoded_outsize,
        .out_format = JPEG_IMAGE_FORMAT_RGB888,
        .out_scale = JPEG_IMAGE_SCALE_1_4,
        .flags = {
            .swap_color_bytes = 0,
        }};
    esp_jpeg_image_output_t outimg;
    esp_jpeg_decode(&jpeg_cfg, &outimg);

    int result = 0;
    int index = 0;
    int global_index = 0;

    // If the model is currently processing, it will only write to the buffer that will be sent via wifi
    // The check for is_processing should be before writing the whole buffer, as if it stops processing
    // we don't want the ML model buffer to be written before a new frame arrives
    if (is_processing)
    {
        for (int row = 0; row < 45; row++)
        {
            for (int i = 0; i < 45; i++)
            {
                index = 18 + i + row * 80; // we only keep a 45x45 image out of the 80x45 image that arrives, and we only keep the red channel
                if (decoded[index * 3] >= CAMERA_THRESHOLD) // frames are binary, red or black
                {
                    result = 1;
                    ml_frame_rgb[global_index * 3] = 255;
                }
                else
                {
                    result = 0;
                    ml_frame_rgb[global_index * 3] = 0;
                }

                global_index += 1;
            }
        }
    }
    else
    {
        for (int row = 0; row < 45; row++)
        {
            for (int i = 0; i < 45; i++)
            {
                index = 18 + i + row * 80;
                if (decoded[index * 3] >= CAMERA_THRESHOLD)
                {
                    result = 1;
                    ml_frame_rgb[global_index * 3] = 255;
                }
                else
                {
                    result = 0;
                    ml_frame_rgb[global_index * 3] = 0;
                }
                // Same as above but we write to the ML model buffer
                int output = dl::tool::round(result * DL_RESCALE(exponent));
                input_ptr[global_index] = DL_CLIP(output, DL_QUANT16_MIN, DL_QUANT16_MAX);

                global_index += 1;
            }
        }
        is_new_image = true;
    }

    free(decoded);

    encode_rgb_to_jpeg(ml_frame_rgb, &ml_frame_jpeg_len, ml_frame_jpeg);
}

// single core implementation of the task that was just above
void gesture_recognition(uvc_frame_t *frame)
{

    uint8_t *decoded;
    int decoded_outsize = 3 * 320 / 4 * 180 / 4;
    decoded = (uint8_t *)malloc(decoded_outsize);

    esp_jpeg_image_cfg_t jpeg_cfg = {
        .indata = (uint8_t *)frame->data,
        .indata_size = frame->data_bytes,
        .outbuf = decoded,
        .outbuf_size = (uint32_t)decoded_outsize,
        .out_format = JPEG_IMAGE_FORMAT_RGB888,
        .out_scale = JPEG_IMAGE_SCALE_1_4,
        .flags = {
            .swap_color_bytes = 0,
        }};

    esp_jpeg_image_output_t outimg;
    esp_jpeg_decode(&jpeg_cfg, &outimg);

    int result = 0;
    int index = 0;
    int global_index = 0;
    for (int row = 0; row < 45; row++)
    {
        for (int i = 0; i < 45; i++)
        {
            index = 18 + i + row * 80;
            if (decoded[index * 3] >= CAMERA_THRESHOLD)
            {
                result = 1;
                ml_frame_rgb[global_index * 3] = 255;
            }
            else
            {
                result = 0;
                ml_frame_rgb[global_index * 3] = 0;
            }
            int output = dl::tool::round(result * DL_RESCALE(exponent));
            input_ptr[global_index] = DL_CLIP(output, DL_QUANT16_MIN, DL_QUANT16_MAX);

            global_index += 1;

        }
    }

    model->run();

    int max_index = std::distance(output_ptr, std::max_element(output_ptr, output_ptr + 8));

    ESP_LOGE(TAG, "Result %s", classes[max_index]);

    free(decoded);

    encode_rgb_to_jpeg(ml_frame_rgb, &ml_frame_jpeg_len, ml_frame_jpeg);
}

// This is the RTOS task associated with gesture recognition
void gesture_recognition_task(void *pvParameters)
{   
    // Initialization of the spi bus. This ESP is the master
    spi_bus_handle_t bus_handle = NULL;
    spi_bus_device_handle_t device_handle = NULL;
    uint8_t data8_out = 0x0;

    spi_config_t bus_conf = {
        .miso_io_num = (gpio_num_t)3,
        .mosi_io_num = (gpio_num_t)4,
        .sclk_io_num = (gpio_num_t)5,
    }; // spi_bus configurations

    spi_device_config_t device_conf = {
        .cs_io_num = (gpio_num_t)6,
        .mode = 0,
        .clock_speed_hz = 20 * 1000, // 20kHz
    }; // spi_device configurations

    bus_handle = spi_bus_create(SPI2_HOST, &bus_conf); // create spi bus
    device_handle = spi_bus_device_create(bus_handle, &device_conf); // create spi device

    while (1)
        {
            
        // Check if a new image is ready and not being processed
        if (is_new_image && !is_processing)
        {
            is_processing = true; // starts processing
            is_new_image = false;

            // Process the latest image
            model->run();
            int max_index = std::distance(output_ptr, std::max_element(output_ptr, output_ptr + 8)); // result of the model

            ESP_LOGE(TAG, "Certainty of prediction : %d", output_ptr[max_index]);

            if (output_ptr[max_index] < 10000) {
                ESP_LOGE(TAG, "Not certain enough, seeting to nothing");
                max_index = 7;
            }

            max_index_byte = static_cast<uint8_t>(max_index);

            data8_out = max_index_byte;
            spi_bus_transfer_bytes(device_handle, &data8_out, NULL, 1); // only write 1 byte with spi device

            ESP_LOGE(TAG, "Result %s", classes[max_index]);

            // Mark processing complete
            is_processing = false;
        }

        // Short delay to allow other tasks to run
        vTaskDelay(pdMS_TO_TICKS(10));
    }

    spi_bus_device_delete(&device_handle);
    spi_bus_delete(&bus_handle);
}

// Help function, encoding rgb888 to jpeg
void encode_rgb_to_jpeg(uint8_t *rgb_data, int *jpeg_size, uint8_t *jpeg_buf)
{
    jpeg_error_t ret = JPEG_ERR_OK;
    int image_size = 3 * 45 * 45;

    int outbuf_size = 3 * 45 * 45;
    // process
    ret = jpeg_enc_process(jpeg_enc, rgb_data, image_size, jpeg_buf, outbuf_size, jpeg_size);
    if (ret != JPEG_ERR_OK)
    {
        ESP_LOGE(TAG, "Error encoding jpeg");
        return;
    }
}