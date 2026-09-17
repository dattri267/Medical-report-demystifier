package com.demystifier.backend.dto;

public class AnalyzeResponse {
    private TextResultDto textResult;
    private VisionResultDto visionResult; // null until an image is uploaded
    private DiscrepancyDto discrepancy; // null if there's nothing to compare yet

    public TextResultDto getTextResult() {
        return textResult;
    }

    public void setTextResult(TextResultDto textResult) {
        this.textResult = textResult;
    }

    public VisionResultDto getVisionResult() {
        return visionResult;
    }

    public void setVisionResult(VisionResultDto visionResult) {
        this.visionResult = visionResult;
    }

    public DiscrepancyDto getDiscrepancy() {
        return discrepancy;
    }

    public void setDiscrepancy(DiscrepancyDto discrepancy) {
        this.discrepancy = discrepancy;
    }
}
