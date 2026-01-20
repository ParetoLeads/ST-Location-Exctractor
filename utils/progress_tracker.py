"""
Progress tracking utilities for KMZ Location Scraper.

Separates concerns: message parsing, state management, and UI updates.
"""
import re
from typing import Dict, List, Optional, Callable, Any
from typing_extensions import TypedDict
import streamlit as st
from config import config


class ProgressState(TypedDict, total=False):
    """Type definition for progress state dictionary."""
    current_stage: str
    stage_progress: int
    locations_found: int
    gpt_batches_completed: int
    gpt_batches_total: int
    hierarchy_batches_completed: int
    hierarchy_batches_total: int
    estimated_time: str
    percent_complete: int
    total_expected_batches: int
    last_progress_value: float
    boundary_points: int


class MessageParser:
    """Parses log messages to extract progress metrics."""
    
    @staticmethod
    def extract_boundary_points(msg: str) -> Optional[int]:
        """Extract number of boundary points from message."""
        match = re.search(r'(\d+)\s+boundary points', msg)
        return int(match.group(1)) if match else None
    
    @staticmethod
    def extract_location_count(msg: str) -> Optional[int]:
        """Extract total location count from summary message."""
        match = re.search(r'found:\s*(\d+)', msg)
        return int(match.group(1)) if match else None
    
    @staticmethod
    def extract_added_locations(msg: str) -> Optional[int]:
        """Extract added location count from intermediate message."""
        match = re.search(r'Added\s+(\d+)\s+locations', msg)
        return int(match.group(1)) if match else None
    
    @staticmethod
    def extract_batch_info(msg: str) -> Optional[tuple]:
        """Extract batch number and total from message.
        
        Returns:
            Tuple of (current_batch, total_batches) or None
        """
        match = re.search(r'batch (\d+)/(\d+)', msg)
        if match:
            return (int(match.group(1)), int(match.group(2)))
        return None
    
    @staticmethod
    def extract_estimated_time(msg: str) -> Optional[str]:
        """Extract estimated remaining time from message."""
        match = re.search(r'time:\s*(.+)$', msg)
        return match.group(1).strip() if match else None
    
    @staticmethod
    def detect_stage(msg: str) -> Optional[tuple]:
        """Detect current processing stage from message.
        
        Returns:
            Tuple of (stage_name, stage_number) or None
        """
        if any(keyword in msg for keyword in ["Extracting boundary", "KMZ", "boundary points", "Parsing KMZ"]):
            return ("Parsing boundary coordinates from KMZ file", 1)
        elif any(keyword in msg for keyword in ["Finding OSM Locations", "Discovering locations"]) or \
             ("OSM" in msg and "Processing" not in msg):
            return ("Discovering locations within boundary", 2)
        elif any(keyword in msg.lower() for keyword in ["Administrative Hierarchy", "hierarchy", "Retrieving administrative"]):
            return ("Retrieving administrative boundaries", 3)
        elif any(keyword in msg.lower() for keyword in ["GPT", "population", "Estimating", "Calculating population"]):
            return ("Calculating population estimates", 4)
        elif any(keyword in msg for keyword in ["Excel", "Saved", "Compiling results"]):
            return ("Compiling results into Excel export", 5)
        return None


class ProgressTracker:
    """Manages progress state and calculations."""
    
    def __init__(self):
        """Initialize progress tracker with default state."""
        self.state: ProgressState = {
            "current_stage": "",
            "stage_progress": 0,
            "locations_found": 0,
            "gpt_batches_completed": 0,
            "gpt_batches_total": 0,
            "hierarchy_batches_completed": 0,
            "hierarchy_batches_total": 0,
            "estimated_time": "",
            "percent_complete": 0,
            "total_expected_batches": 0,
            "last_progress_value": 0.0
        }
        self.parser = MessageParser()
        self.total_stages = config.TOTAL_STAGES
    
    def update_from_message(self, msg: str) -> None:
        """Update progress state based on log message."""
        # Extract boundary points
        boundary_points = self.parser.extract_boundary_points(msg)
        if boundary_points is not None:
            self.state["boundary_points"] = boundary_points
        
        # Extract location count
        location_count = self.parser.extract_location_count(msg)
        if location_count is not None:
            self.state["locations_found"] = location_count
            # Calculate expected batches
            batch_size = config.DEFAULT_BATCH_SIZE
            expected_batches_per_phase = (location_count + batch_size - 1) // batch_size
            self.state["total_expected_batches"] = expected_batches_per_phase * 2
            self.state["hierarchy_batches_total"] = expected_batches_per_phase
            self.state["gpt_batches_total"] = expected_batches_per_phase
        else:
            # Check for intermediate "Added X locations" messages
            added_count = self.parser.extract_added_locations(msg)
            if added_count is not None:
                self.state["locations_found"] = self.state.get("locations_found", 0) + added_count
        
        # Extract batch information
        if "Retrieving hierarchy batch" in msg or "Processing hierarchy batch" in msg:
            batch_info = self.parser.extract_batch_info(msg)
            if batch_info:
                current, total = batch_info
                self.state["hierarchy_batches_completed"] = current
                self.state["hierarchy_batches_total"] = total
                # Update total expected batches
                gpt_total = self.state.get("gpt_batches_total", 0)
                if gpt_total > 0:
                    self.state["total_expected_batches"] = total + gpt_total
                elif self.state.get("total_expected_batches", 0) == 0:
                    self.state["total_expected_batches"] = total * 2
        
        if "Processing GPT batch" in msg or "Calculating population for batch" in msg:
            batch_info = self.parser.extract_batch_info(msg)
            if batch_info:
                current, total = batch_info
                self.state["gpt_batches_completed"] = current
                self.state["gpt_batches_total"] = total
                # Update total expected batches
                hierarchy_total = self.state.get("hierarchy_batches_total", 0)
                if hierarchy_total > 0:
                    self.state["total_expected_batches"] = hierarchy_total + total
                elif self.state.get("total_expected_batches", 0) == 0:
                    self.state["total_expected_batches"] = total * 2
        
        # Extract estimated time
        estimated_time = self.parser.extract_estimated_time(msg)
        if estimated_time:
            self.state["estimated_time"] = estimated_time
        
        # Detect stage
        stage_info = self.parser.detect_stage(msg)
        if stage_info:
            stage_name, stage_num = stage_info
            self.state["current_stage"] = stage_name
            self.state["stage_progress"] = stage_num
        
        # Calculate percentage complete
        self._calculate_percentage()
    
    def _calculate_percentage(self) -> None:
        """Calculate overall percentage complete."""
        hierarchy_done = self.state.get("hierarchy_batches_completed", 0)
        gpt_done = self.state.get("gpt_batches_completed", 0)
        hierarchy_total = self.state.get("hierarchy_batches_total", 0)
        gpt_total = self.state.get("gpt_batches_total", 0)
        total_expected = self.state.get("total_expected_batches", 0)
        
        # Calculate total batches
        done_batches = hierarchy_done + gpt_done
        
        if hierarchy_total > 0 and gpt_total > 0:
            total_batches = hierarchy_total + gpt_total
        elif hierarchy_total > 0:
            total_batches = hierarchy_total * 2
        elif gpt_total > 0:
            total_batches = gpt_total * 2
        elif total_expected > 0:
            total_batches = total_expected
        else:
            total_batches = 0
        
        if total_batches > 0:
            # Leave 10% for KMZ/OSM parsing and 10% for Excel export
            batch_percent = (done_batches / total_batches) * config.PROGRESS_BATCH_PERCENT
            new_percent = config.PROGRESS_KMZ_OSM_PERCENT + batch_percent
            # Ensure progress never decreases
            self.state["percent_complete"] = max(
                self.state.get("percent_complete", 0),
                int(new_percent)
            )
    
    def get_progress_value(self) -> float:
        """Get progress value for progress bar (0.0 to 1.0)."""
        percent = self.state.get("percent_complete", 0)
        if percent == 0:
            # Use stage-based progress if no batch progress yet
            stage_progress = self.state.get("stage_progress", 0)
            percent = (stage_progress / self.total_stages) * 100
        
        # Ensure progress never decreases - always use the maximum
        current_progress = self.state.get("last_progress_value", 0.0)
        progress_value = max(current_progress, min(percent / 100, 0.99))  # Cap at 99% until complete
        self.state["last_progress_value"] = progress_value
        return progress_value


class ProgressUI:
    """Handles UI updates for progress display."""
    
    def __init__(
        self,
        stage_text_container: Any,
        progress_bar: Any,
        metrics_container: Any,
        status_text_container: Any
    ):
        """Initialize progress UI with Streamlit containers.
        
        Args:
            stage_text_container: Streamlit empty container for stage text
            progress_bar: Streamlit progress bar
            metrics_container: Streamlit empty container for metrics
            status_text_container: Streamlit empty container for status text
        """
        self.stage_text = stage_text_container
        self.progress_bar = progress_bar
        self.metrics = metrics_container
        self.status_text = status_text_container
    
    def update(self, tracker: ProgressTracker, message: str) -> None:
        """Update all UI elements with current progress state.
        
        Args:
            tracker: ProgressTracker instance with current state
            message: Current log message
        """
        state = tracker.state
        
        # Update progress bar
        progress_value = tracker.get_progress_value()
        self.progress_bar.progress(progress_value)
        
        # Update stage text
        with self.stage_text:
            percent_display = state.get("percent_complete", 0)
            est_time = state.get("estimated_time", "")
            stage_info = state.get("current_stage", "Initializing...")
            
            if percent_display > 0 and est_time:
                st.markdown(f"### {stage_info} — {percent_display}% complete (Est: {est_time})")
            elif percent_display > 0:
                st.markdown(f"### {stage_info} — {percent_display}% complete")
            elif est_time:
                st.markdown(f"### {stage_info} (Est: {est_time})")
            else:
                st.markdown(f"### {stage_info}")
        
        # Update metrics
        with self.metrics:
            metrics_parts = []
            if state.get("locations_found", 0) > 0:
                metrics_parts.append(f"{state['locations_found']} locations")
            if state.get("hierarchy_batches_total", 0) > 0:
                metrics_parts.append(
                    f"Hierarchy: {state['hierarchy_batches_completed']}/{state['hierarchy_batches_total']}"
                )
            if state.get("gpt_batches_total", 0) > 0:
                metrics_parts.append(
                    f"GPT: {state['gpt_batches_completed']}/{state['gpt_batches_total']}"
                )
            if state.get("estimated_time"):
                metrics_parts.append(f"Est: {state['estimated_time']}")
            
            if metrics_parts:
                st.caption(" | ".join(metrics_parts))
        
        # Update status text
        with self.status_text:
            st.text(message)
    
    def mark_complete(self) -> None:
        """Mark progress as complete."""
        self.progress_bar.progress(1.0)
        with self.stage_text:
            st.markdown("### Analysis complete!")
        with self.metrics:
            pass  # Clear metrics
        with self.status_text:
            st.text("Analysis complete!")


def create_progress_callback(
    tracker: ProgressTracker,
    ui: ProgressUI,
    progress_messages: List[str]
) -> Callable[[str], None]:
    """
    Create a progress callback function for use with LocationAnalyzer.
    
    Args:
        tracker: ProgressTracker instance
        ui: ProgressUI instance
        progress_messages: List to append messages to
        
    Returns:
        Callback function that takes a message string
    """
    def callback(msg: str) -> None:
        progress_messages.append(msg)
        tracker.update_from_message(msg)
        ui.update(tracker, msg)
    
    return callback
