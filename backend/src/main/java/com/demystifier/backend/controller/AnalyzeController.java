package com.demystifier.backend.controller;

import com.demystifier.backend.dto.AnalyzeRequest;
import com.demystifier.backend.dto.AnalyzeResponse;
import com.demystifier.backend.service.TextEngineClient;
import org.springframework.web.bind.annotation.*;

@RestController
@RequestMapping("/analyze")
public class AnalyzeController {

    private final TextEngineClient textEngineClient;

    public AnalyzeController(TextEngineClient textEngineClient) {
        this.textEngineClient = textEngineClient;
    }

    @PostMapping
    public AnalyzeResponse analyze(@RequestBody AnalyzeRequest request) {
        AnalyzeResponse response = new AnalyzeResponse();

        if (request.getReportText() != null && !request.getReportText().isBlank()) {
            response.setTextResult(textEngineClient.summarize(request.getReportText()));
        }

        // vision + discrepancy logic comes in Steps 6-7 later
        return response;
    }
}