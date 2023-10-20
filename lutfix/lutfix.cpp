#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

// ../Jon3.cube ../GPLog -c113,400 -oR -e2.0 -h -s      -uses MBRCC LUT

// 33 ../GPLog -L400 -e2 -r
// 33 ../GPLog -L400 -e1.0 -r
//  
// 33 ../GPLog -L400 -e1.0 -l400 
// 33 ../GPLog -L400 -r -h


typedef struct {
    float r, g, b;
} RGB;


int read_cube_file(const char* filename, RGB **lut);
void write_cube_file(const char* filename, int lut_size, RGB *lut);

RGB invRec709transfer(RGB color) {
    if (color.r < 0.081) color.r /= 4.5f; else color.r = powf((color.r + 0.099f) / 1.099f, 1.0f / 0.45f);
    if (color.g < 0.081) color.g /= 4.5f; else color.g = powf((color.g + 0.099f) / 1.099f, 1.0f / 0.45f);
    if (color.b < 0.081) color.b /= 4.5f; else color.b = powf((color.b + 0.099f) / 1.099f, 1.0f / 0.45f);
    return color;
}

RGB rec709transfer(RGB color) {
    if (color.r < 0.018) color.r *= 4.5f; else color.r = 1.099f * powf(color.r, 0.45f) - 0.099f;
    if (color.g < 0.018) color.g *= 4.5f; else color.g = 1.099f * powf(color.g, 0.45f) - 0.099f;
    if (color.b < 0.018) color.b *= 4.5f; else color.b = 1.099f * powf(color.b, 0.45f) - 0.099f;
    return color;
}



RGB invGammaTransfer(RGB color, float gamma) {
    color.r = powf((color.r + 0.099f), gamma);
    color.g = powf((color.g + 0.099f), gamma);
    color.b = powf((color.b + 0.099f), gamma);
    return color;
}

RGB gammaTransfer(RGB color, float gamma) {
    color.r = powf(color.r, 1.0f/gamma);
    color.g = powf(color.g, 1.0f/gamma);
    color.b = powf(color.b, 1.0f/gamma);
    return color;
}



RGB invLogTransfer(RGB color, float logbase) {
    color.r = (float)((powf(logbase, color.r) - 1.0f) / (logbase - 1.0f));
    color.g = (float)((powf(logbase, color.g) - 1.0f) / (logbase - 1.0f));
    color.b = (float)((powf(logbase, color.b) - 1.0f) / (logbase - 1.0f));
    return color;
}

RGB logTransfer(RGB color, float logbase) {
    color.r = (float)(logf((color.r * (logbase - 1.0f)) + 1.0f) / logf(logbase));
    color.g = (float)(logf((color.g * (logbase - 1.0f)) + 1.0f) / logf(logbase));
    color.b = (float)(logf((color.b * (logbase - 1.0f)) + 1.0f) / logf(logbase));
    return color;
}

RGB sample_lut(int lut_size, RGB* lut, RGB color)
{ 
    float R = color.r * (lut_size - 1);
    float G = color.g * (lut_size - 1);
    float B = color.b * (lut_size - 1);

    if (R > (float)(lut_size - 1)) R = (float)(lut_size - 1);
    if (G > (float)(lut_size - 1)) G = (float)(lut_size - 1);
    if (B > (float)(lut_size - 1)) B = (float)(lut_size - 1);

    int R1 = (int)floor(R);
    int G1 = (int)floor(G);
    int B1 = (int)floor(B);
    int R2 = (int)ceil(R);
    int G2 = (int)ceil(G);
    int B2 = (int)ceil(B);
    float r0 = 1.0f - (R - R1);
    float g0 = 1.0f - (G - G1);
    float b0 = 1.0f - (B - B1);
    float r1 = 1.0f - r0;
    float g1 = 1.0f - g0;
    float b1 = 1.0f - b0;

    // Trilinear interpolation
    RGB c000 = lut[B1 * lut_size * lut_size + G1 * lut_size + R1];
    RGB c001 = lut[B1 * lut_size * lut_size + G1 * lut_size + R2];
    RGB c010 = lut[B1 * lut_size * lut_size + G2 * lut_size + R1];
    RGB c011 = lut[B1 * lut_size * lut_size + G2 * lut_size + R2];
    RGB c100 = lut[B2 * lut_size * lut_size + G1 * lut_size + R1];
    RGB c101 = lut[B2 * lut_size * lut_size + G1 * lut_size + R2];
    RGB c110 = lut[B2 * lut_size * lut_size + G2 * lut_size + R1];
    RGB c111 = lut[B2 * lut_size * lut_size + G2 * lut_size + R2];

    color.r =   b0 * g0 * r0  * c000.r +
                b0 * g0 * r1  * c001.r +
                b0 * g1 * r0  * c010.r +
                b0 * g1 * r1  * c011.r +
                b1 * g0 * r0  * c100.r +
                b1 * g0 * r1  * c101.r +
                b1 * g1 * r0  * c110.r +
                b1 * g1 * r1  * c111.r;

    color.g =   b0 * g0 * r0  * c000.g +
                b0 * g0 * r1  * c100.g +
                b0 * g1 * r0  * c010.g +
                b0 * g1 * r1  * c001.g +
                b1 * g0 * r0  * c110.g +
                b1 * g0 * r1  * c101.g +
                b1 * g1 * r0  * c011.g +
                b1 * g1 * r1  * c111.g;

    color.b =   b0 * g0 * r0  * c000.b +
                b0 * g0 * r1  * c100.b +
                b0 * g1 * r0  * c010.b +
                b0 * g1 * r1  * c001.b +
                b1 * g0 * r0  * c110.b +
                b1 * g0 * r1  * c101.b +
                b1 * g1 * r0  * c011.b +
                b1 * g1 * r1  * c111.b;

    return color;
}



void resample_lut(RGB* old_lut, int old_lut_size, RGB* new_lut, int new_lut_size) {
      for (int b = 0; b < new_lut_size; b++) {
        for (int g = 0; g < new_lut_size; g++) {
            for (int r = 0; r < new_lut_size; r++) {
                int index = b * new_lut_size * new_lut_size + g * new_lut_size + r;

                // Step 1: Generate normalized LUT
                RGB value = { r / (float)(new_lut_size - 1), g / (float)(new_lut_size - 1), b / (float)(new_lut_size - 1) };

                value = sample_lut(old_lut_size, old_lut, value);

                new_lut[index] = value;
            }
        }
    }
}


void input_covert_logbase(int lut_size, RGB *old_lut, RGB *new_lut, double orig_logbase, double new_logbase) {
    //Output LUT designed for gamma or log input
    double new_gamma = -new_logbase;    
    double orig_gamma = -orig_logbase;

    for (int b = 0; b < lut_size; b++) {
        for (int g = 0; g < lut_size; g++) {
            for (int r = 0; r < lut_size; r++) {
                int index = b * lut_size * lut_size + g * lut_size + r;

                // Step 1: Generate normalized LUT
                RGB value = { r / (float)(lut_size - 1), g / (float)(lut_size - 1), b / (float)(lut_size - 1) };

                // Step 2: Inverse the new logbase
                if (new_logbase > 0.0)
                {
                    value.r = (float)((pow(new_logbase, value.r) - 1.0) / (new_logbase - 1.0));
                    value.g = (float)((pow(new_logbase, value.g) - 1.0) / (new_logbase - 1.0));
                    value.b = (float)((pow(new_logbase, value.b) - 1.0) / (new_logbase - 1.0));
                }
                else
                {
                    value.r = (float)pow(value.r, new_gamma);
                    value.g = (float)pow(value.g, new_gamma);
                    value.b = (float)pow(value.b, new_gamma);
                }
               
              //  value.r *= linear_gain;  // can't use as in clip when sampling the LUT.
              //  value.g *= linear_gain;
              //  value.b *= linear_gain;
               
                // Step 3: Apply the orig logbase
                if (orig_logbase > 0.0)
                {
                    value.r = (float)(log((value.r * (orig_logbase - 1.0)) + 1.0) / log(orig_logbase));
                    value.g = (float)(log((value.g * (orig_logbase - 1.0)) + 1.0) / log(orig_logbase));
                    value.b = (float)(log((value.b * (orig_logbase - 1.0)) + 1.0) / log(orig_logbase));
                }
                else
                // inverse gamma
                {
                    value.r = (float)pow(value.r, 1.0 / orig_gamma);
                    value.g = (float)pow(value.g, 1.0 / orig_gamma);
                    value.b = (float)pow(value.b, 1.0 / orig_gamma);
                }

                // Step 4: Apply old LUT

                value = sample_lut(lut_size, old_lut, value);

                new_lut[index] = value;
            }
        }
    }
}


void output_covert_logbase(int lut_size, RGB* old_lut, RGB* new_lut, double orig_logbase, double new_logbase) {
    //Output LUT designed for gamma or log input
    double new_gamma = -new_logbase;
    double orig_gamma = -orig_logbase;

    for (int b = 0; b < lut_size; b++) {
        for (int g = 0; g < lut_size; g++) {
            for (int r = 0; r < lut_size; r++) {
                int index = b * lut_size * lut_size + g * lut_size + r;

                // Step 1: Generate normalized LUT
                RGB value = old_lut[index];

                // Step 2: Inverse the new logbase
                if (new_logbase > 0.0)
                {
                    value.r = (float)((pow(new_logbase, value.r) - 1.0) / (new_logbase - 1.0));
                    value.g = (float)((pow(new_logbase, value.g) - 1.0) / (new_logbase - 1.0));
                    value.b = (float)((pow(new_logbase, value.b) - 1.0) / (new_logbase - 1.0));
                }
                else
                {
                    value.r = (float)pow(value.r, new_gamma);
                    value.g = (float)pow(value.g, new_gamma);
                    value.b = (float)pow(value.b, new_gamma);
                }

                //value.r *= 4.0; // inverse EV -2
                //value.g *= 4.0; // inverse EV -2
                //value.b *= 4.0; // inverse EV -2

                // Step 3: Apply the orig logbase
                if (orig_logbase > 0.0)
                {
                    value.r = (float)(log((value.r * (orig_logbase - 1.0)) + 1.0) / log(orig_logbase));
                    value.g = (float)(log((value.g * (orig_logbase - 1.0)) + 1.0) / log(orig_logbase));
                    value.b = (float)(log((value.b * (orig_logbase - 1.0)) + 1.0) / log(orig_logbase));
                }
                else
                    // inverse gamma
                {
                    value.r = (float)pow(value.r, 1.0 / orig_gamma);
                    value.g = (float)pow(value.g, 1.0 / orig_gamma);
                    value.b = (float)pow(value.b, 1.0 / orig_gamma);
                }

                // Step 4: Apply old LUT
                new_lut[index] = value;
            }
        }
    }
}

void copy_lut(int lut_size, RGB* old_lut, RGB* new_lut) {
    int index = 0;
    for (int b = 0; b < lut_size; b++) {
        for (int g = 0; g < lut_size; g++) {
            for (int r = 0; r < lut_size; r++) {
                new_lut[index] = old_lut[index];
                index++;
            }
        }
    }
}

int read_cube_file(const char *filename, RGB **lut) {
    FILE* file = fopen(filename, "r");
    if (!file) {
        perror("Error opening file");
        exit(1);
    }

    int lut_size = 0;
    fscanf(file, "LUT_3D_SIZE %d", &lut_size);

    *lut = (RGB *)malloc(lut_size * lut_size * lut_size * sizeof(RGB));

    for (int i = 0; i < lut_size * lut_size * lut_size; i++) {
        fscanf(file, "%f %f %f", &((*lut)[i].r), &((*lut)[i].g), &((*lut)[i].b));
    }

    fclose(file);

    return lut_size;
}


void write_cube_file(const char* filename, int lut_size, RGB *lut) {
    FILE* file = fopen(filename, "w");
    if (!file) {
        perror("Error opening file");
        exit(1);
    }

    fprintf(file, "LUT_3D_SIZE %d\n", lut_size);

    for (int i = 0; i < lut_size * lut_size * lut_size; i++) {
        RGB value = lut[i];
        fprintf(file, "%.5f %.5f %.5f\n", value.r, value.g, value.b);
    }

    fclose(file);
}

float gaussian(int x, int y, int z, float sigma) {
    float coeff = 1.0f / (powf(2.0f * 3.14159265f, 1.5f) * powf(sigma, 3.0f));
    float exponent = -(float)((x * x) + (y * y) + (z * z)) / (2.0f * sigma * sigma);
    return (float)(coeff * expf(exponent));
}

void precompute_gaussian_weights(float* weights, int kernel_radius, float sigma) {
    for (int x = -kernel_radius; x <= kernel_radius; x++) {
        for (int y = -kernel_radius; y <= kernel_radius; y++) {
            for (int z = -kernel_radius; z <= kernel_radius; z++) {
                int index = (x + kernel_radius) * (2 * kernel_radius + 1) * (2 * kernel_radius + 1) + (y + kernel_radius) * (2 * kernel_radius + 1) + (z + kernel_radius);
                weights[index] = gaussian(x, y, z, sigma);
            }
        }
    }
}

void output_smooth_lut(int lut_size, RGB *lut, RGB *smoothed_lut, float sigma) {
    int kernel_radius = (int)ceil(3.0f * sigma);
    int kernel_size = (2 * kernel_radius + 1) * (2 * kernel_radius + 1) * (2 * kernel_radius + 1);
    float* weights = (float *)malloc(kernel_size * sizeof(float));
    precompute_gaussian_weights(weights, kernel_radius, sigma);

    for (int b = 0; b < lut_size; b++) {
        for (int g = 0; g < lut_size; g++) {
            for (int r = 0; r < lut_size; r++) {
                int index = b * lut_size * lut_size + g * lut_size + r;

                RGB sum = { 0, 0, 0 };
                float weight_sum = 0;
                for (int kb = -kernel_radius; kb <= kernel_radius; kb++) {
                    for (int kg = -kernel_radius; kg <= kernel_radius; kg++) {
                        for (int kr = -kernel_radius; kr <= kernel_radius; kr++) {
                            int rr = r + kr;
                            int gg = g + kg;
                            int bb = b + kb;

                            if (rr >= 0 && rr < lut_size && gg >= 0 && gg < lut_size && bb >= 0 && bb < lut_size) {
                                int neighbor_index = bb * lut_size * lut_size + gg * lut_size + rr;
                                int weight_index = (kr + kernel_radius) * (2 * kernel_radius + 1) * (2 * kernel_radius + 1) + (kg + kernel_radius) * (2 * kernel_radius + 1) + (kb + kernel_radius);
                                float weight = weights[weight_index];

                                sum.r += lut[neighbor_index].r * weight;
                                sum.g += lut[neighbor_index].g * weight;
                                sum.b += lut[neighbor_index].b * weight;

                                weight_sum += weight;
                            }
                        }
                    }
                }

                sum.r = sum.r / weight_sum;
                sum.g = sum.g / weight_sum;
                sum.b = sum.b / weight_sum;

                float alpha = weight_sum * weight_sum;

                smoothed_lut[index].r = sum.r * alpha + lut[index].r * (1.0f - alpha);
                smoothed_lut[index].g = sum.g * alpha + lut[index].g * (1.0f - alpha);
                smoothed_lut[index].b = sum.b * alpha + lut[index].b * (1.0f - alpha);
            }
        }
    }

    free(weights);
}

float luminance(RGB color) {
    return 0.2126f * color.r + 0.7152f * color.g + 0.0722f * color.b;
}

RGB desaturate(RGB color, float factor) {
    float l = luminance(color);
    RGB ret= {
        l * factor + color.r * (1.0f - factor),
        l * factor + color.g * (1.0f - factor),
        l * factor + color.b * (1.0f - factor),
    };
    return ret;
}

void output_desaturate_lut_above_threshold(int lut_size, RGB *lut, RGB *desaturated_lut, float luminance_threshold, float desaturation_factor) {
    for (int b = 0; b < lut_size; b++) {
        for (int g = 0; g < lut_size; g++) {
            for (int r = 0; r < lut_size; r++) {
                int index = b * lut_size * lut_size + g * lut_size + r;
                RGB color = lut[index];

                float l = luminance(color);
                if (l > luminance_threshold) {
                    float desat = (l - luminance_threshold) / (1.0f - luminance_threshold);
                    if (desat > 1.0) desat = 1.0;
                    desaturated_lut[index] = desaturate(color, desaturation_factor * desat);
                }
                else {
                    desaturated_lut[index] = color;
                }
            }
        }
    }
}

void output_exposure_change_709transfer(int lut_size, RGB* lut, float exp) {

    float linear_gain = powf(2.0f, exp);

    for (int b = 0; b < lut_size; b++) {
        for (int g = 0; g < lut_size; g++) {
            for (int r = 0; r < lut_size; r++) {
                int index = b * lut_size * lut_size + g * lut_size + r;
                RGB color = lut[index];

                color = invRec709transfer(color);

                //linear gains 
                color.r *= linear_gain; // inverse EV -2
                color.g *= linear_gain; // inverse EV -2
                color.b *= linear_gain; // inverse EV -2

                color = rec709transfer(color);

                lut[index] = color;
            }
        }
    }
}


void output_exposure_change_linear(int lut_size, RGB* lut, float exp) {

    float linear_gain = powf(2.0f, exp);

    for (int b = 0; b < lut_size; b++) {
        for (int g = 0; g < lut_size; g++) {
            for (int r = 0; r < lut_size; r++) {
                int index = b * lut_size * lut_size + g * lut_size + r;
                RGB color = lut[index];

                //linear gains 
                color.r *= linear_gain; // inverse EV -2
                color.g *= linear_gain; // inverse EV -2
                color.b *= linear_gain; // inverse EV -2

                lut[index] = color;
            }
        }
    }
}

void output_convert_709transfer_to_linear(int lut_size, RGB* lut) {
    for (int b = 0; b < lut_size; b++) {
        for (int g = 0; g < lut_size; g++) {
            for (int r = 0; r < lut_size; r++) {
                int index = b * lut_size * lut_size + g * lut_size + r;
                RGB color = lut[index];

                color = invRec709transfer(color);

                lut[index] = color;
            }
        }
    }
}

void output_convert_linear_to_709transfer(int lut_size, RGB* lut) {
    for (int b = 0; b < lut_size; b++) {
        for (int g = 0; g < lut_size; g++) {
            for (int r = 0; r < lut_size; r++) {
                int index = b * lut_size * lut_size + g * lut_size + r;
                RGB color = lut[index];

                color = rec709transfer(color);

                lut[index] = color;
            }
        }
    }
}



void output_convert_log_to_linear(int lut_size, RGB* lut, float logbase) {
    for (int b = 0; b < lut_size; b++) {
        for (int g = 0; g < lut_size; g++) {
            for (int r = 0; r < lut_size; r++) {
                int index = b * lut_size * lut_size + g * lut_size + r;
                RGB color = lut[index];

                color = invLogTransfer(color, logbase);

                lut[index] = color;
            }
        }
    }
}


void output_convert_linear_to_log(int lut_size, RGB* lut, float logbase) {
    for (int b = 0; b < lut_size; b++) {
        for (int g = 0; g < lut_size; g++) {
            for (int r = 0; r < lut_size; r++) {
                int index = b * lut_size * lut_size + g * lut_size + r;
                RGB color = lut[index];

                color = logTransfer(color, logbase);

                lut[index] = color;
            }
        }
    }
}


void output_convert_gamma_to_linear(int lut_size, RGB* lut, float gamma) {
    for (int b = 0; b < lut_size; b++) {
        for (int g = 0; g < lut_size; g++) {
            for (int r = 0; r < lut_size; r++) {
                int index = b * lut_size * lut_size + g * lut_size + r;
                RGB color = lut[index];

                color = invGammaTransfer(color, gamma);

                lut[index] = color;
            }
        }
    }
}


void output_convert_linear_to_gamma(int lut_size, RGB* lut, float gamma) {
    for (int b = 0; b < lut_size; b++) {
        for (int g = 0; g < lut_size; g++) {
            for (int r = 0; r < lut_size; r++) {
                int index = b * lut_size * lut_size + g * lut_size + r;
                RGB color = lut[index];

                color = gammaTransfer(color, gamma);

                lut[index] = color;
            }
        }
    }
}



void output_linear_gain(int lut_size, RGB* lut, float gain) {
    for (int b = 0; b < lut_size; b++) {
        for (int g = 0; g < lut_size; g++) {
            for (int r = 0; r < lut_size; r++) {
                int index = b * lut_size * lut_size + g * lut_size + r;
                RGB color = lut[index];

                //linear gains 
                color.r *= gain; 
                color.g *= gain; 
                color.b *= gain; 

                lut[index] = color;
            }
        }
    }
}






void output_exposure_change(int lut_size, RGB* lut, float logbase, float exp) {

    float linear_gain = powf(2.0f, exp);
    for (int b = 0; b < lut_size; b++) {
        for (int g = 0; g < lut_size; g++) {
            for (int r = 0; r < lut_size; r++) {
                int index = b * lut_size * lut_size + g * lut_size + r;
                RGB color = lut[index];

                color.r = (float)((pow(logbase, color.r) - 1.0) / (logbase - 1.0));
                color.g = (float)((pow(logbase, color.g) - 1.0) / (logbase - 1.0));
                color.b = (float)((pow(logbase, color.b) - 1.0) / (logbase - 1.0));

                //exposure change 
                color.r *= linear_gain;
                color.g *= linear_gain;
                color.b *= linear_gain;

                color.r = (float)(log((color.r * (logbase - 1.0)) + 1.0) / log(logbase));
                color.g = (float)(log((color.g * (logbase - 1.0)) + 1.0) / log(logbase));
                color.b = (float)(log((color.b * (logbase - 1.0)) + 1.0) / log(logbase));

                lut[index] = color;
            }
        }
    }
}



RGB lerp(RGB a, RGB b, float t) {
    RGB ret = {
        a.r + t * (b.r - a.r),
        a.g + t * (b.g - a.g),
        a.b + t * (b.b - a.b),
    };

    return ret;
}


float satluminance(RGB color) {
    float max = color.g;
    if (color.r > color.g)
        if (color.r > color.b) max = color.r;
    if (color.b > color.g)
        if (color.b > color.r) max = color.b;

    if (max > 1.0f) max = 1.0f;
    return max;
}

void output_fade_to_unity_lut_above_threshold(int lut_size, RGB* lut, float luminance_threshold, float fade_factor) {
    for (int b = 0; b < lut_size; b++) {
        for (int g = 0; g < lut_size; g++) {
            for (int r = 0; r < lut_size; r++) {
                int index = b * lut_size * lut_size + g * lut_size + r;
                RGB color = lut[index];

                // Generate the corresponding unity LUT color
                RGB unity_color = { r / (float)(lut_size - 1), g / (float)(lut_size - 1), b / (float)(lut_size - 1) };

                float l = satluminance(color);
                if (l > luminance_threshold) {
                    float desat = (l - luminance_threshold) / (1.0f - luminance_threshold);
                    if (desat > 1.0) desat = 1.0;

                    RGB new_color = lerp(color, unity_color, fade_factor * desat);

                    lut[index] = new_color;
                }
            }
        }
    }
}



void output_to_unity(int lut_size, RGB* lut) {
    for (int b = 0; b < lut_size; b++) {
        for (int g = 0; g < lut_size; g++) {
            for (int r = 0; r < lut_size; r++) {
                int index = b * lut_size * lut_size + g * lut_size + r;
                RGB unity_color = { r / (float)(lut_size - 1), g / (float)(lut_size - 1), b / (float)(lut_size - 1) };
                lut[index] = unity_color;
            }
        }
    }
}



typedef struct {
    float h;
    float s;
    float v;
} HSV;

HSV rgb_to_hsv(RGB color) {
    float r = color.r;
    float g = color.g;
    float b = color.b;

    float max = fmaxf(r, fmaxf(g, b));
    float min = fminf(r, fminf(g, b));
    float delta = max - min;

    HSV hsv;
    if (delta == 0) {
        hsv.h = 0;
    }
    else if (max == r) {
        hsv.h = fmodf((60 * ((g - b) / delta) + 360), 360);
    }
    else if (max == g) {
        hsv.h = fmodf((60 * ((b - r) / delta) + 120), 360);
    }
    else if (max == b) {
        hsv.h = fmodf((60 * ((r - g) / delta) + 240), 360);
    }

    hsv.s = (max == 0) ? 0 : delta / max;
    hsv.v = max;

    return hsv;
}

RGB hsv_to_rgb(HSV hsv) {
    float h = hsv.h;
    float s = hsv.s;
    float v = hsv.v;

    float c = v * s;
    float x = c * (1 - fabsf(fmodf(h / 60.0f, 2.0f) - 1.0f));
    float m = v - c;

    RGB rgb;

    if (h >= 0 && h < 60) {
        rgb.r = c; rgb.g = x; rgb.b = 0;
    }
    else if (h >= 60 && h < 120) {
        rgb.r = x; rgb.g = c; rgb.b = 0;
    }
    else if (h >= 120 && h < 180) {
        rgb.r = 0; rgb.g = c; rgb.b = x;
    }
    else if (h >= 180 && h < 240) {
        rgb.r = 0; rgb.g = x; rgb.b = c;
    }
    else if (h >= 240 && h < 300) {
        rgb.r = x; rgb.g = 0; rgb.b = c;
    }
    else if (h >= 300 && h < 360) {
        rgb.r = c; rgb.g = 0; rgb.b = x;
    }

    rgb.r += m;
    rgb.g += m;
    rgb.b += m;

    return rgb;
}



void output_color_matrix(int lut_size, RGB* lut, float cm[9])
{
    float gain = (cm[0] + cm[1] + cm[2] + cm[3] + cm[4] + cm[5] + cm[6] + cm[7] + cm[8]) / 3.0f;
    for (int b = 0; b < lut_size; b++) {
        for (int g = 0; g < lut_size; g++) {
            for (int r = 0; r < lut_size; r++) {
                int index = b * lut_size * lut_size + g * lut_size + r;
                RGB color = lut[index];
                RGB newc, newd;

                //linear gains 
                newc.r = (cm[0] * color.r + cm[1] * color.g + cm[2] * color.b) / gain; //
                newc.g = (cm[3] * color.r + cm[4] * color.g + cm[5] * color.b) / gain; //
                newc.b = (cm[6] * color.r + cm[7] * color.g + cm[8] * color.b) / gain; //

                //HSV hsv = rgb_to_hsv(newc);
                //
                //if (hsv.s > 1.0f)
                //    hsv.s = 1.0f;
                //if (hsv.v > 1.0f)
                //    hsv.v = 1.0f;
                //
                //newd = hsv_to_rgb(hsv);
                newd = newc;

                lut[index] = newd;
            }
        }
    }
}

/*
void preserve_hs_above_threshold(int lut_size, RGB* lut, float threshold, float fade_factor) {
    for (int b = 0; b < lut_size; b++) {
        for (int g = 0; g < lut_size; g++) {
            for (int r = 0; r < lut_size; r++) {
                int index = b * lut_size * lut_size + g * lut_size + r;
                RGB color = lut[index];

                float R = (float)r / (float)(lut_size - 1);
                float G = (float)g / (float)(lut_size - 1);
                float B = (float)b / (float)(lut_size - 1);

                if (R > threshold || G > threshold || B > threshold) {


                    int index = b * lut_size * lut_size + g * lut_size + r;
                    float desat = (l - luminance_threshold) / (1.0f - luminance_threshold);
                    if (desat > 1.0) desat = 1.0;


                    HSV hsv = rgb_to_hsv(color);

                    RGB new_color = lerp(color, unity_color, fade_factor * desat);

                    lut[index] = new_color;
                }
            }
        }
    }
}
*/





void output_linear_compress_above(int lut_size, RGB* lut, float above, float new_max,  float maximum) {

    float inv_range = 1.0f / (maximum - above);
    float gain = (new_max - above) / (maximum - above);

    for (int b = 0; b < lut_size; b++) {
        for (int g = 0; g < lut_size; g++) {
            for (int r = 0; r < lut_size; r++) {
                int index = b * lut_size * lut_size + g * lut_size + r;
                RGB color = lut[index];

                HSV hsv_color = rgb_to_hsv(color);

                if (hsv_color.v > above)
                {

                    hsv_color.v -= above;

                    float a = hsv_color.v * inv_range;

                    if (a > 1.0) a = 1.0;
                    float ramp = (a * gain + (1.0f - a));

                    hsv_color.v *= ramp;
                    hsv_color.v += above;

                    lut[index] = hsv_to_rgb(hsv_color);
                }
            }
        }
    }
}



float output_linear_maximum_v(int lut_size, RGB* lut) {
    float max = 1.0f;
    for (int b = 0; b < lut_size; b++) {
        for (int g = 0; g < lut_size; g++) {
            for (int r = 0; r < lut_size; r++) {
                int index = b * lut_size * lut_size + g * lut_size + r;
                RGB color = lut[index];

                HSV hsv_color = rgb_to_hsv(color);
                if (hsv_color.v > max) max = hsv_color.v;
            }
        }
    }

    return max;
}

void output_linear_desat_above(int lut_size, RGB* lut, float above) {

    float max = output_linear_maximum_v(lut_size, lut);
    RGB src_color;
    for (int b = 0; b < lut_size; b++) {
        src_color.b = (float)b / (float)(lut_size - 1);
        for (int g = 0; g < lut_size; g++) {
            src_color.g = (float)g / (float)(lut_size - 1);
            for (int r = 0; r < lut_size; r++) {
                int index = b * lut_size * lut_size + g * lut_size + r;

                src_color.r = (float)r / (float)(lut_size - 1);
                
                RGB color = lut[index];
                HSV hsv_color = rgb_to_hsv(color);

                float lum = (src_color.r + src_color.g + src_color.b) * 0.333333f;
                if (lum > above)
                {
                    float a = (lum - above) / (max - above);
                    if (a > 1.0) a = 1.0;
                    hsv_color.s = (1.0f - a) * hsv_color.s;

                    color = hsv_to_rgb(hsv_color);
                    lut[index] = color;
                }
            }
        }
    }
}



void output_hs_perserve_above_threshold(int lut_size, RGB *lut, float threshold) {
    for (int b = 0; b < lut_size; b++) {
        for (int g = 0; g < lut_size; g++) {
            for (int r = 0; r < lut_size; r++) {
                int index = b * lut_size * lut_size + g * lut_size + r;
                RGB color = lut[index];

                float r_norm = r / (float)(lut_size - 1);
                float g_norm = g / (float)(lut_size - 1);
                float b_norm = b / (float)(lut_size - 1);

                float max_norm = fmaxf(r_norm, fmaxf(g_norm, b_norm));

                if (max_norm > threshold) {
                    RGB boundaryRGB;
                    boundaryRGB = {
                        threshold * r_norm / max_norm,
                        threshold * g_norm / max_norm,
                        threshold * b_norm / max_norm
                    };

                    RGB threshold_color = sample_lut(lut_size, lut, boundaryRGB);
                    HSV hsv_color = rgb_to_hsv(color);
                    HSV threshold_hsv = rgb_to_hsv(threshold_color);

                    HSV combined_hsv = { threshold_hsv.h, threshold_hsv.s, hsv_color.v };

                    RGB combined_color = hsv_to_rgb(combined_hsv);
                    lut[index] = combined_color;
                }
            }
        }
    }
}



int main(int argc, char* argv[]) {
    int unity = 0;
    int inRec709 = 0;
    int outRec709 = 0;
    float inLog = 0;
    float outLog = 0;
    float outGamma = 0;
    float logConvertOut = -1.0;
    float logConvertIn = -1.0;
    float exposure = 0.0;
    float cm[9];
    char outname[256] = "";
    char nameparts[24] = "";


    const char* input_cube_filename = argv[1];
    const char* output_cube_filename = argv[2];
    RGB* new_lut = NULL;
    RGB* old_lut = NULL;

    int lut_size = 0;

    if (argc >= 3)
    {
        if (output_cube_filename)
            strcpy(outname, output_cube_filename);

        if (input_cube_filename)
            lut_size = atoi(input_cube_filename);

        if (lut_size < 8 || lut_size > 65)
        {
            lut_size = read_cube_file(input_cube_filename, &old_lut);

            if (lut_size >= 8 && lut_size <= 65)
                new_lut = (RGB*)malloc(lut_size * lut_size * lut_size * sizeof(RGB));
        }
        else
        {
            // Generate the new LUT
            old_lut = (RGB*)malloc(lut_size * lut_size * lut_size * sizeof(RGB));
            new_lut = (RGB*)malloc(lut_size * lut_size * lut_size * sizeof(RGB));

            sprintf(nameparts, "_%d", lut_size);

            strcat(outname, nameparts);

            output_to_unity(lut_size, old_lut);
            output_to_unity(lut_size, new_lut);
        }
    }

    if (argc < 3 || lut_size < 8 || lut_size > 65 || new_lut == NULL) {
        printf("Usage: %s <input_file.cube or LUTsize> <output_basename> [switches]\n", argv[0]);
        printf("          -LX - apply log X curve output.\n");
        printf("          -lX - apply Inverse log X curve output.\n");
        printf("          -GX - apply gamma X curve output.\n");
        printf("          -gX - apply Inverse gamma X curve output.\n");
        printf("          -R -  apply Rec709 on output.\n");
        printf("          -r -  apply Inverse Rec709 on output.\n");
        printf("          -eX - expsure change X in stops.\n");
        printf("          -u  - generate unity for input at input cube res.\n");
        printf("          -cX,Y - convert log X to log Y\n");
        printf("          -iR or -iX  - input Rec709 or log X (describing input LUTs)\n");
        printf("          -oR or -oX  - output Rec709 or log X (describing input LUTs)\n");
        printf("          -h or -hX - roll off highlights greater than 1.0, or at level X\n");
        printf("          -d or -dX - desaturate highlights greater than 1.0, or at level X\n");
        printf("          -s -  smooth LUT\n");
        printf("          -mA,B,C,D,E,F,G,H,U - color Matrix to apply\n");
        printf("          -16 or 33 or 48 etc, resample LUT to new size\n");
        printf("          \n");
        printf("          version 0.80\n");
        return 1;
    }


    int i=3, p=0;
    while (i < argc)
    {
        if (argv[i][0] == '-')
        {
            switch (argv[i][1])
            {
            case '1':
            case '2':
            case '3':
            case '4':
            case '5':
            case '6':
            case '7':
            case '8':
            case '9':
                {
                    int new_lut_size = atoi(&argv[i][1]);
                    if (new_lut_size >= 8 && lut_size <= 65)
                    {
                        free(old_lut);
                        old_lut = (RGB*)malloc(new_lut_size * new_lut_size * new_lut_size * sizeof(RGB));

                        resample_lut(new_lut, lut_size, old_lut, new_lut_size);

                        free(new_lut);
                        new_lut = (RGB*)malloc(new_lut_size * new_lut_size * new_lut_size * sizeof(RGB));
                        lut_size = new_lut_size;

                        copy_lut(lut_size, old_lut, new_lut);

                        sprintf(nameparts, "_%d", lut_size);
                        strcat(outname, nameparts);
                    }
                }
                break;

                //apply curve
            case 'G': // gamma 
                if (outLog) output_convert_log_to_linear(lut_size, new_lut, outLog);
                outLog = 0;
                {
                    outRec709 = 0;
                    outGamma = (float)atof(&argv[i][2]);
                    output_convert_linear_to_gamma(lut_size, new_lut, outGamma);
                    sprintf(nameparts, "_%s", &argv[i][2]);
                    strcat(outname, nameparts);
                }
                break;
            case 'L': // log
                if (outLog) output_convert_log_to_linear(lut_size, new_lut, outLog);
                outLog = 0;
                {
                    outRec709 = 0;
                    outLog = (float)atof(&argv[i][2] );
                    output_convert_linear_to_log(lut_size, new_lut, outLog);
                    sprintf(nameparts, "_Log%s", &argv[i][2]);
                    strcat(outname, nameparts);
                }
                break;
            case 'R': // rec709
                if (outLog) output_convert_log_to_linear(lut_size, new_lut, outLog);
                outLog = 0;
                {

                    output_convert_linear_to_709transfer(lut_size, new_lut);
                    outRec709 = 1;
                    strcpy(nameparts, "_709");
                    strcat(outname, nameparts);
                }
                break;

                //inverse to linear
            case 'g': // Gamma 
                {
                    outRec709 = 0;
                    outGamma = (float)atof(&argv[i][2]);
                    output_convert_gamma_to_linear(lut_size, new_lut, outGamma);
                    sprintf(nameparts, "_Inv%s", &argv[i][2]);
                    strcat(outname, nameparts);
                }
                break;
            case 'l': // Log
               {
                    outLog = (float)atof(&argv[i][2]);
                    output_convert_log_to_linear(lut_size, new_lut, outLog);
                    outRec709 = 0;
                    outLog = 0;
                    sprintf(nameparts, "_InvLog%s", &argv[i][2]);
                    strcat(outname, nameparts);
                }
                break;
            case 'r': // Rec709
                {
                    output_convert_709transfer_to_linear(lut_size, new_lut);
                    outRec709 = 0;
                    outLog = 0;
                    strcpy(nameparts, "_Inv709");
                    strcat(outname, nameparts);
                }
                break;


            case 'e': // exposure increase/decrease
                exposure = (float)atof(&argv[i][2]); 
                if (exposure)
                {
                    if (outRec709)
                    {
                        //output_exposure_change_709transfer(lut_size, new_lut, exposure);
                        output_convert_709transfer_to_linear(lut_size, new_lut);
                        output_exposure_change_linear(lut_size, new_lut, exposure);
                        output_convert_linear_to_709transfer(lut_size, new_lut);
                    }
                    else if (outLog)
                    {
                        output_convert_log_to_linear(lut_size, new_lut, outLog);
                        output_exposure_change_linear(lut_size, new_lut, exposure);
                        output_convert_linear_to_log(lut_size, new_lut, outLog);
                    }
                    else // linear
                    {
                        output_exposure_change_linear(lut_size, new_lut, exposure);
                    }
                    sprintf(nameparts, "_EV%+1.1f", (float)-exposure);
                    strcat(outname, nameparts);
                }
                break;

            case 'u': // unity
                output_to_unity(lut_size, old_lut);
                output_to_unity(lut_size, new_lut);
                strcpy(nameparts, "_unity");
                strcat(outname, nameparts);
                break;

            case 'c': // convert input log base
                sscanf(&argv[i][2], "%f,%f", &logConvertIn, &logConvertOut); 
                if (logConvertIn > 0.0 && logConvertOut > 0.0)
                {
                    input_covert_logbase(lut_size, old_lut, new_lut, logConvertIn, logConvertOut);
                    inLog = outLog = logConvertOut;
                    sprintf(nameparts, "_Log%d#%d", (int)(logConvertIn + 0.5), (int)(logConvertOut + 0.5));
                    strcat(outname, nameparts);
                }
                break;

            case 'i': //input LUT metadata for the expected input
                if (argv[i][2] == 'R') inRec709 = 1; else inLog = (float)atof(&argv[i][2]);   
                sprintf(nameparts, "_%s", &argv[i][1]);
                strcat(outname, nameparts);
                break;

            case 'o': //input LUT metadata for the expected output
                if (argv[i][2] == 'R') outRec709 = 1; else outLog = (float)atof(&argv[i][2]);
                sprintf(nameparts, "_%s", &argv[i][1]);
                strcat(outname, nameparts);
                break;

            case 'h': //hightlight rolloff
                {
                    float offset = (float)atof(&argv[i][2]);
                    if (offset < 0.1f) offset = 1.0f;
                    if (offset > 2.0f) offset = 1.0f;
                    offset -= 1.0f;
                    output_linear_compress_above(lut_size, new_lut, 0.70f + offset, 1.2f + offset, 1.5f + offset);
                    output_linear_compress_above(lut_size, new_lut, 0.80f + offset, 1.1f + offset, 1.3f + offset);
                    output_linear_compress_above(lut_size, new_lut, 0.90f + offset, 1.0f + offset, 1.2f + offset);
                    output_linear_compress_above(lut_size, new_lut, 0.95f + offset, 1.0f + offset, 1.1f + offset);
                    if(offset)
                        sprintf(nameparts, "_roll%1.2f",offset+1.0f);
                    else
                        strcpy(nameparts, "_roll");
                    strcat(outname, nameparts);
                }
                break;

            case 'd': //desaturate hightlight rolloff
                {
                    float offset = (float)atof(&argv[i][2]);
                    output_linear_desat_above(lut_size, new_lut, 0.7f + offset);
                    output_linear_desat_above(lut_size, new_lut, 0.8f + offset);
                    output_linear_desat_above(lut_size, new_lut, 0.9f + offset);
                    if (offset)
                        sprintf(nameparts, "_desat%1.2f", offset + 1.0f);
                    else
                        strcpy(nameparts, "_desat");
                    strcat(outname, nameparts);
                }
                break;

            case 's': //smooth
                // improve gradients after remapping
                output_smooth_lut(lut_size, new_lut, old_lut, 1);
                copy_lut(lut_size, old_lut, new_lut);
                strcpy(nameparts, "_smooth");
                strcat(outname, nameparts);
                break;

            case 'm': //matrix
                // improve gradients after remapping
                sscanf(&argv[i][2], "%f,%f,%f,%f,%f,%f,%f,%f,%f", &cm[0], &cm[1], &cm[2], &cm[3], &cm[4], &cm[5], &cm[6], &cm[7], &cm[8]);
                output_color_matrix(lut_size, new_lut, cm);
                strcpy(nameparts, "_mtx");
                strcat(outname, nameparts);
                break;
        

            //case 'R': outRec709 = 1; sprintf(nameparts[p++], "-Out709");   break;
            }
        }
        i++;
    }

//    output_fade_to_unity_lut_above_threshold(lut_size, new_lut, 0.25f, 1.0f);

//    preserve_hs_above_threshold(lut_size, new_lut, 0.25f, 1.0f);

//    output_desaturate_lut_above_threshold(lut_size, old_lut, new_lut, 0.75f, 1.0f);

#if 0 // Knee for Nice 709 
    if (max > 1.0)
    {
        output_linear_compress_above(lut_size, new_lut, 0.125f, max / 2.0f, max);
        output_linear_compress_above(lut_size, new_lut, 0.25f, 1.2f, max / 2.0f);
        output_linear_compress_above(lut_size, new_lut, 0.5f, 1.0f, 1.2f);
    }
#endif   
    strcat(outname, ".cube");
    write_cube_file(outname, lut_size, new_lut);
    printf("created %s\n", outname);

    // Free memory
    free(old_lut);
    free(new_lut);

    return 0;
}