package com.demystifier.backend.dto;

import java.util.List;

public class TextResultDto {
    private String plainLanguageSummary;
    private List<String> keyFindings;
    private String disclaimer;

    public TextResultDto() {
    }

    public TextResultDto(String plainLanguageSummary, List<String> keyFindings, String disclaimer) {
        this.plainLanguageSummary = plainLanguageSummary;
        this.keyFindings = keyFindings;
        this.disclaimer = disclaimer;
    }

    public String getPlainLanguageSummary() {
        return plainLanguageSummary;
    }

    public void setPlainLanguageSummary(String plainLanguageSummary) {
        this.plainLanguageSummary = plainLanguageSummary;
    }

    public List<String> getKeyFindings() {
        return keyFindings;
    }

    public void setKeyFindings(List<String> keyFindings) {
        this.keyFindings = keyFindings;
    }

    public String getDisclaimer() {
        return disclaimer;
    }

    public void setDisclaimer(String disclaimer) {
        this.disclaimer = disclaimer;
    }
}