import logging
import time
from typing import Any, Dict, Optional, List

# Import optional dependencies
try:
    # Import essential OpenAI Agents tracing classes
    from agents.tracing import (
        add_trace_processor,
        set_trace_processors,
        Span,
        Trace,
        TraceProcessor
    )
    AGENTS_TRACING_AVAILABLE = True
except ImportError:
    AGENTS_TRACING_AVAILABLE = False
    # Create dummy classes for type checking
    class Span: pass
    class Trace: pass
    class TraceProcessor: pass

logger = logging.getLogger(__name__)

class RagMetricsTraceProcessor:
    """
    A trace processor for OpenAI Agents that sends traces to RagMetrics.
    
    This processor implements the OpenAI Agents TraceProcessor interface
    and forwards traces to RagMetrics for monitoring and visualization.
    """
    
    def __init__(self, rm_client: Any):
        """
        Initialize the trace processor with a RagMetricsClient.
        
        Args:
            rm_client: An instance of RagMetricsClient
        """
        self.rm_client = rm_client
        self.spans_processed = 0
        self.traces_processed = 0
        logger.info("RagMetricsTraceProcessor initialized")
    
    def process_span(self, span: Span) -> None:
        """
        Process a span by logging it to RagMetrics.
        
        Args:
            span: The span to process
        """
        if not self.rm_client or self.rm_client.logging_off:
            logger.debug(f"Skipping span processing: client logging is off")
            return
        
        try:
            # Extract span data in a format suitable for RagMetrics
            span_data = {
                "trace_id": span.trace_id,
                "span_id": span.id,
                "name": span.name,
                "parent_id": span.parent_id,
                "started_at": span.started_at,
                "ended_at": span.ended_at,
                "duration": (span.ended_at - span.started_at).total_seconds() if span.ended_at else 0
            }
            
            # Add span data if available
            if hasattr(span, "span_data"):
                if hasattr(span.span_data, "model_dump"):
                    span_data["data"] = span.span_data.model_dump()
                else:
                    span_data["data"] = str(span.span_data)
            
            # Log this span to RagMetrics
            self.rm_client._log_trace(
                input_messages={"span_type": span.name, "trace_id": span.trace_id},
                response=span_data,
                metadata_llm={
                    "source": "openai_agents_tracing",
                    "span_type": span.name,
                    "integration": "openai_agents",
                    "trace_processor": "RagMetricsTraceProcessor"
                },
                duration=span_data["duration"],
                conversation_id=span.trace_id  # Use trace_id as conversation_id for grouping
            )
            
            self.spans_processed += 1
            logger.debug(f"Processed span: {span.name} (total: {self.spans_processed})")
        except Exception as e:
            logger.error(f"Error processing span {span.name}: {e}", exc_info=True)
    
    def process_trace(self, trace: Trace) -> None:
        """
        Process a completed trace by logging it to RagMetrics.
        
        Args:
            trace: The trace to process
        """
        if not self.rm_client or self.rm_client.logging_off:
            logger.debug(f"Skipping trace processing: client logging is off")
            return
        
        try:
            # Extract trace data
            trace_data = {
                "trace_id": trace.trace_id,
                "workflow_name": trace.workflow_name,
                "group_id": trace.group_id,
                "spans_count": len(trace.spans) if hasattr(trace, "spans") else 0,
                "duration": (trace.end_time - trace.start_time).total_seconds() if hasattr(trace, "end_time") and trace.end_time else 0
            }
            
            # Log summary of this trace to RagMetrics
            self.rm_client._log_trace(
                input_messages={"trace_type": trace.workflow_name, "trace_id": trace.trace_id},
                response=trace_data,
                metadata_llm={
                    "source": "openai_agents_tracing",
                    "type": "trace_summary",
                    "integration": "openai_agents",
                    "trace_processor": "RagMetricsTraceProcessor"
                },
                duration=trace_data["duration"],
                conversation_id=trace.trace_id  # Use trace_id as conversation_id for grouping
            )
            
            self.traces_processed += 1
            logger.info(f"Processed trace: {trace.workflow_name} with {trace_data['spans_count']} spans (total traces: {self.traces_processed})")
        except Exception as e:
            logger.error(f"Error processing trace {trace.workflow_name}: {e}", exc_info=True)

def register_trace_processor(rm_client):
    """
    Register the RagMetricsTraceProcessor with OpenAI Agents tracing.
    
    Args:
        rm_client: An instance of RagMetricsClient
        
    Returns:
        bool: True if registration was successful, False otherwise
    """
    if not AGENTS_TRACING_AVAILABLE:
        logger.warning("OpenAI Agents tracing is not available. Cannot register trace processor.")
        return False
    
    try:
        # Create the processor
        processor = RagMetricsTraceProcessor(rm_client)
        
        # Add it to the OpenAI Agents tracing system
        add_trace_processor(processor)
        
        logger.info("RagMetricsTraceProcessor successfully registered with OpenAI Agents")
        return True
    except Exception as e:
        logger.error(f"Failed to register RagMetricsTraceProcessor: {e}", exc_info=True)
        return False 