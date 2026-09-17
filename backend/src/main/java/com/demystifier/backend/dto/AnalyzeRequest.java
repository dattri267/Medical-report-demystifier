package com.demystifier.backend.dto;

public class AnalyzeRequest {
    private String reportText;   // optional — null if only an image was uploaded

    public String getReportText() { return reportText; }
    public void setReportText(String reportText) { this.reportText = reportText; }
}