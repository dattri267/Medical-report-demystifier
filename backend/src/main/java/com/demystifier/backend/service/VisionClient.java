package com.demystifier.backend.service;

import com.demystifier.backend.dto.VisionResultDto;
import org.springframework.web.multipart.MultipartFile;

public interface VisionClient {
    VisionResultDto analyzeImage(MultipartFile image);
}