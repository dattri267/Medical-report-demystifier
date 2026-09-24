package com.demystifier.backend.repository;

import com.demystifier.backend.entity.ReviewCase;
import org.springframework.data.jpa.repository.JpaRepository;

import java.util.List;

public interface ReviewCaseRepository extends JpaRepository<ReviewCase, Long> {
    List<ReviewCase> findByReviewedFalseOrderByCreatedAtDesc();

    List<ReviewCase> findByPriorityOrderByCreatedAtDesc(String priority);
}