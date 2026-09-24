package com.demystifier.backend.controller;

import com.demystifier.backend.dto.AnalyzeResponse;
import com.demystifier.backend.dto.VisionResultDto;
import com.demystifier.backend.service.TextEngineClient;
import com.demystifier.backend.service.VisionClient;
import org.springframework.web.bind.annotation.*;
import org.springframework.web.multipart.MultipartFile;

@RestController
@RequestMapping("/analyze")
public class AnalyzeController {

    private final TextEngineClient textEngineClient;
    private final VisionClient visionClient;

    public AnalyzeController(TextEngineClient textEngineClient, VisionClient visionClient) {
        this.textEngineClient = textEngineClient;
        this.visionClient = visionClient;
    }

    @PostMapping(consumes = "multipart/form-data")
    public AnalyzeResponse analyzeMultipart(
            @RequestParam(value = "reportText", required = false) String reportText,
            @RequestParam(value = "image", required = false) MultipartFile image) {

        AnalyzeResponse response = new AnalyzeResponse();

        if (reportText != null && !reportText.isBlank()) {
            response.setTextResult(textEngineClient.summarize(reportText));
        }

        if (image != null && !image.isEmpty()) {
            VisionResultDto visionResult = visionClient.analyzeImage(image);
            response.setVisionResult(visionResult);
        }

        return response;
    }

    // Keep the original JSON-only endpoint for pure text testing (Postman
    // convenience)
    @PostMapping(consumes = "application/json")
    public AnalyzeResponse analyzeJson(@RequestBody com.demystifier.backend.dto.AnalyzeRequest request) {
        AnalyzeResponse response = new AnalyzeResponse();
        if (request.getReportText() != null && !request.getReportText().isBlank()) {
            response.setTextResult(textEngineClient.summarize(request.getReportText()));
        }
        return response;
    }
}