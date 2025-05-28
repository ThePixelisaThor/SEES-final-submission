#ifndef GESTURE_RECOGNITION_HPP
#define GESTURE_RECOGNITION_HPP

// most ESP libraries are written in C
extern "C" {
    #include <stdio.h>
    #include <string.h>
    #include <assert.h>
    #include <inttypes.h>
    #include "freertos/FreeRTOS.h"
    #include "freertos/event_groups.h"
    #include "freertos/task.h"
    #include "esp_err.h"
    #include "esp_log.h"
    #include "usb_stream.h"
    #include "jpeg_decoder.h"
    #include "esp_jpeg_common.h"
    #include "esp_jpeg_enc.h"
    #include "i2c_bus.h"
    #include "spi_bus.h"
}

// Those libraries are written in c++
#include "dl_model_base.hpp"
#include "dl_tensor_base.hpp"

void push_frame(uvc_frame_t *frame);
void gesture_recognition_task(void *pvParameters);
// Unused functions
// static void print_red_channel_ascii(uint8_t *rgb_image, int width, int height);
// static void print_red_channel_ascii2(uint8_t *rgb_image, int width, int height);
void gesture_recognition(uvc_frame_t *frame);
void initialize_model_jpeg_spi();

uint8_t get_result();
uint8_t* get_rgb_frame();
uint8_t* get_jpeg_frame();
size_t get_jpeg_len();

void encode_rgb_to_jpeg(uint8_t *rgb_data, int *jpeg_size, uint8_t *jpeg_buf);

#define CAMERA_THRESHOLD 128 // a simple threshold works

#define I2C_MASTER_SCL_IO   (gpio_num_t)15       /*!< gpio number for I2C master clock */
#define I2C_MASTER_SDA_IO   (gpio_num_t)16       /*!< gpio number for I2C master data  */
#define I2C_MASTER_FREQ_HZ  100000               /*!< I2C master clock frequency */
#define ESP_SLAVE_ADDR      0x28                 /*!< ESP32 slave address, you can set any 7bit value */

#endif // GESTURE_RECOGNITION_HPP
