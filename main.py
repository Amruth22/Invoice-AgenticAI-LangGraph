"""
Main Streamlit Application for Invoice Processing with LangGraph
AI-powered invoice automation system with agentic workflows
"""

import os
import asyncio
import pandas as pd
import streamlit as st
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go
from typing import Dict, Any, List

# Import our LangGraph workflow and components
from graph import get_workflow
from state import ProcessingStatus, ValidationStatus, RiskLevel, PaymentStatus
from utils.logger import setup_logging, get_logger

# Configure Streamlit page
st.set_page_config(
    page_title="Invoice AgenticAI - LangGraph",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize logging
setup_logging(log_level="INFO", log_file="logs/invoice_system.log")
logger = get_logger("main_app")

# Custom CSS for better UI
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: white;
        padding: 1rem;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        border-left: 4px solid #667eea;
    }
    .status-success { color: #28a745; font-weight: bold; }
    .status-warning { color: #ffc107; font-weight: bold; }
    .status-error { color: #dc3545; font-weight: bold; }
    .status-info { color: #17a2b8; font-weight: bold; }
</style>
""", unsafe_allow_html=True)


class InvoiceProcessingApp:
    """Main Streamlit application class"""
    
    def __init__(self):
        self.workflow = None
        self.initialize_session_state()
    
    def initialize_session_state(self):
        """Initialize Streamlit session state"""
        if 'processing_results' not in st.session_state:
            st.session_state.processing_results = []
        if 'workflow_initialized' not in st.session_state:
            st.session_state.workflow_initialized = False
        if 'last_refresh' not in st.session_state:
            st.session_state.last_refresh = datetime.now()
    
    def initialize_workflow(self):
        """Initialize the LangGraph workflow"""
        if not st.session_state.workflow_initialized:
            with st.spinner("Initializing AI agents and workflow..."):
                try:
                    # Configuration for agents
                    config = {
                        "document_agent": {
                            "extraction_methods": ["pymupdf", "pdfplumber"],
                            "ai_confidence_threshold": 0.7
                        },
                        "validation_agent": {
                            "po_file_path": "data/purchase_orders.csv",
                            "fuzzy_threshold": 80,
                            "amount_tolerance": 0.05
                        },
                        "risk_agent": {
                            "risk_thresholds": {
                                "low": 0.3, "medium": 0.6, "high": 0.8, "critical": 0.9
                            }
                        },
                        "payment_agent": {
                            "payment_api_url": "http://localhost:8000/initiate_payment",
                            "auto_payment_threshold": 5000,
                            "manual_approval_threshold": 25000
                        }
                    }
                    
                    self.workflow = get_workflow(config)
                    st.session_state.workflow_initialized = True
                    st.success("✅ AI agents and workflow initialized successfully!")
                    
                except Exception as e:
                    st.error(f"❌ Failed to initialize workflow: {str(e)}")
                    logger.error(f"Workflow initialization failed: {e}")
                    return False
        else:
            self.workflow = get_workflow()
        
        return True
    
    def render_header(self):
        """Render the main header"""
        st.markdown("""
        <div class="main-header">
            <h1>🤖 Invoice AgenticAI - LangGraph</h1>
            <p>AI-Powered Invoice Processing with Intelligent Agent Workflows</p>
        </div>
        """, unsafe_allow_html=True)
    
    def render_sidebar(self):
        """Render the sidebar with controls"""
        st.sidebar.title("🎛️ Control Panel")
        
        # Workflow configuration
        st.sidebar.subheader("Workflow Configuration")
        workflow_type = st.sidebar.selectbox(
            "Workflow Type",
            ["standard", "high_value", "expedited"],
            help="Select the type of workflow to use for processing"
        )
        
        priority_level = st.sidebar.slider(
            "Priority Level",
            min_value=1, max_value=5, value=1,
            help="1=Low, 5=Critical"
        )
        
        max_concurrent = st.sidebar.slider(
            "Max Concurrent Processing",
            min_value=1, max_value=10, value=3,
            help="Maximum number of invoices to process simultaneously"
        )
        
        # File selection
        st.sidebar.subheader("📁 Invoice Files")
        invoice_files = self.get_available_files()
        
        if invoice_files:
            selected_files = st.sidebar.multiselect(
                "Select invoices to process",
                invoice_files,
                default=invoice_files[:3] if len(invoice_files) >= 3 else invoice_files
            )
        else:
            st.sidebar.warning("No invoice files found in data/invoices/")
            selected_files = []
        
        # Processing controls
        st.sidebar.subheader("🚀 Processing Controls")
        
        col1, col2 = st.sidebar.columns(2)
        with col1:
            process_button = st.button("🔄 Process Invoices", type="primary")
        with col2:
            clear_button = st.button("🗑️ Clear Results")
        
        if clear_button:
            st.session_state.processing_results = []
            st.rerun()
        
        # System status
        st.sidebar.subheader("🔍 System Status")
        if st.sidebar.button("Health Check"):
            self.show_health_check()
        
        return {
            "workflow_type": workflow_type,
            "priority_level": priority_level,
            "max_concurrent": max_concurrent,
            "selected_files": selected_files,
            "process_button": process_button
        }
    
    def get_available_files(self) -> List[str]:
        """Get list of available invoice files"""
        invoice_dir = "data/invoices"
        if not os.path.exists(invoice_dir):
            os.makedirs(invoice_dir, exist_ok=True)
            return []
        
        files = [f for f in os.listdir(invoice_dir) if f.lower().endswith('.pdf')]
        return sorted(files)
    
    async def process_invoices_async(self, selected_files: List[str], 
                                   workflow_type: str, priority_level: int,
                                   max_concurrent: int):
        """Process invoices asynchronously"""
        if not selected_files:
            st.warning("Please select at least one invoice file to process.")
            return
        
        # Create progress tracking
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        try:
            # Process invoices
            config = {"priority_level": priority_level}
            
            status_text.text(f"Processing {len(selected_files)} invoices...")
            
            results = await self.workflow.process_batch(
                selected_files, 
                workflow_type=workflow_type,
                max_concurrent=max_concurrent
            )
            
            # Update progress
            progress_bar.progress(1.0)
            status_text.text("✅ Processing completed!")
            
            # Store results in session state
            st.session_state.processing_results = results
            st.session_state.last_refresh = datetime.now()
            
            # Show summary
            self.show_processing_summary(results)
            
        except Exception as e:
            st.error(f"❌ Processing failed: {str(e)}")
            logger.error(f"Invoice processing failed: {e}")
        finally:
            progress_bar.empty()
            status_text.empty()
    
    def show_processing_summary(self, results: List):
        """Show processing summary"""
        if not results:
            return
        
        # Calculate summary statistics
        total = len(results)
        completed = sum(1 for r in results if r.overall_status == ProcessingStatus.COMPLETED)
        escalated = sum(1 for r in results if r.overall_status == ProcessingStatus.ESCALATED)
        failed = sum(1 for r in results if r.overall_status == ProcessingStatus.FAILED)
        
        # Display summary metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("📊 Total Processed", total)
        with col2:
            st.metric("✅ Completed", completed, delta=f"{completed/total*100:.1f}%")
        with col3:
            st.metric("⚠️ Escalated", escalated, delta=f"{escalated/total*100:.1f}%")
        with col4:
            st.metric("❌ Failed", failed, delta=f"{failed/total*100:.1f}%")
        
        st.success(f"🎉 Processing completed! {completed}/{total} invoices processed successfully.")
    
    def render_main_dashboard(self):
        """Render the main dashboard"""
        if not st.session_state.processing_results:
            st.info("👆 Select invoice files from the sidebar and click 'Process Invoices' to get started!")
            
            # Show sample workflow diagram
            self.show_workflow_diagram()
            return
        
        # Create tabs for different views
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "📊 Overview", "📋 Invoice Details", "🔍 Agent Performance", 
            "⚠️ Escalations", "📈 Analytics"
        ])
        
        with tab1:
            self.render_overview_tab()
        
        with tab2:
            self.render_invoice_details_tab()
        
        with tab3:
            self.render_agent_performance_tab()
        
        with tab4:
            self.render_escalations_tab()
        
        with tab5:
            self.render_analytics_tab()
    
    def render_overview_tab(self):
        """Render overview tab"""
        results = st.session_state.processing_results
        
        if not results:
            st.info("No processing results available.")
            return
        
        # Status distribution
        status_counts = {}
        for result in results:
            status = result.overall_status.value
            status_counts[status] = status_counts.get(status, 0) + 1
        
        # Create status chart
        fig_status = px.pie(
            values=list(status_counts.values()),
            names=list(status_counts.keys()),
            title="Processing Status Distribution"
        )
        st.plotly_chart(fig_status, width='stretch')
        
        # Processing timeline
        timeline_data = []
        for result in results:
            timeline_data.append({
                "Invoice": result.file_name,
                "Start": result.created_at,
                "End": result.updated_at,
                "Duration": (result.updated_at - result.created_at).total_seconds() / 60,
                "Status": result.overall_status.value
            })
        
        if timeline_data:
            df_timeline = pd.DataFrame(timeline_data)
            
            fig_timeline = px.bar(
                df_timeline,
                x="Invoice",
                y="Duration",
                color="Status",
                title="Processing Duration by Invoice"
            )
            fig_timeline.update_layout(xaxis_tickangle=-45)
            st.plotly_chart(fig_timeline, width='stretch')
    
    def render_invoice_details_tab(self):
        """Render invoice details tab"""
        results = st.session_state.processing_results
        
        if not results:
            st.info("No processing results available.")
            return
        
        # Create detailed table
        invoice_data = []
        for result in results:
            row = {
                "File": result.file_name,
                "Status": result.overall_status.value,
                "Invoice #": result.invoice_data.invoice_number if result.invoice_data else "N/A",
                "Customer": result.invoice_data.customer_name if result.invoice_data else "N/A",
                "Amount": f"${result.invoice_data.total:.2f}" if result.invoice_data else "N/A",
                "Risk Level": result.risk_assessment.risk_level.value if result.risk_assessment else "N/A",
                "Payment Status": result.payment_decision.payment_status.value if result.payment_decision else "N/A",
                "Processing Time": f"{(result.updated_at - result.created_at).total_seconds() / 60:.1f} min"
            }
            invoice_data.append(row)
        
        df_invoices = pd.DataFrame(invoice_data)
        
        # Add status styling
        def style_status(val):
            if val == "completed":
                return "background-color: #d4edda; color: #155724"
            elif val == "escalated":
                return "background-color: #fff3cd; color: #856404"
            elif val == "failed":
                return "background-color: #f8d7da; color: #721c24"
            return ""
        
        styled_df = df_invoices.style.map(style_status, subset=['Status'])
        st.dataframe(styled_df, use_container_width=True)
        
        # Detailed view selector
        st.subheader("🔍 Detailed Invoice View")
        selected_invoice = st.selectbox(
            "Select invoice for detailed view:",
            options=range(len(results)),
            format_func=lambda x: f"{results[x].file_name} - {results[x].invoice_data.customer_name if results[x].invoice_data else 'Unknown'}"
        )
        
        if selected_invoice is not None:
            self.show_detailed_invoice_view(results[selected_invoice])
    
    def show_detailed_invoice_view(self, result):
        """Show detailed view of a single invoice processing result"""
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📄 Invoice Information")
            if result.invoice_data:
                st.write(f"**Invoice Number:** {result.invoice_data.invoice_number}")
                st.write(f"**Customer:** {result.invoice_data.customer_name}")
                st.write(f"**Amount:** ${result.invoice_data.total:.2f}")
                st.write(f"**Due Date:** {result.invoice_data.due_date or 'N/A'}")
                st.write(f"**Extraction Confidence:** {result.invoice_data.extraction_confidence:.2f}")
        
        with col2:
            st.subheader("🎯 Processing Results")
            st.write(f"**Overall Status:** {result.overall_status.value}")
            if result.validation_result:
                st.write(f"**Validation:** {result.validation_result.validation_status.value}")
            if result.risk_assessment:
                st.write(f"**Risk Level:** {result.risk_assessment.risk_level.value}")
                st.write(f"**Risk Score:** {result.risk_assessment.risk_score:.2f}")
            if result.payment_decision:
                st.write(f"**Payment Status:** {result.payment_decision.payment_status.value}")
        
        # Audit trail
        st.subheader("📋 Audit Trail")
        if result.audit_trail:
            audit_data = []
            for entry in result.audit_trail:
                # Format details - extract key info from dict
                details_str = ""
                if entry.details:
                    if isinstance(entry.details, dict):
                        # Extract important fields
                        key_fields = ['duration_ms', 'success_rate', 'execution_count', 'compliance_status', 'reportable_events']
                        details_parts = [f"{k}: {entry.details[k]}" for k in key_fields if k in entry.details]
                        details_str = ", ".join(details_parts) if details_parts else ""
                    else:
                        details_str = str(entry.details)

                audit_data.append({
                    "Timestamp": entry.timestamp.strftime("%Y-%m-%d %H:%M:%S"),
                    "Agent": entry.agent_name,
                    "Action": entry.action,
                    "Status": entry.status.value,
                    "Details": details_str
                })

            df_audit = pd.DataFrame(audit_data)
            st.dataframe(df_audit, use_container_width=True)
    
    def render_agent_performance_tab(self):
        """Render agent performance tab"""
        results = st.session_state.processing_results
        
        if not results:
            st.info("No processing results available.")
            return
        
        # Aggregate agent metrics
        agent_stats = {}
        for result in results:
            for agent_name, metrics in result.agent_metrics.items():
                if agent_name not in agent_stats:
                    agent_stats[agent_name] = {
                        "executions": 0,
                        "successes": 0,
                        "failures": 0,
                        "total_duration": 0
                    }
                
                agent_stats[agent_name]["executions"] += metrics.executions
                agent_stats[agent_name]["successes"] += metrics.successes
                agent_stats[agent_name]["failures"] += metrics.failures
                agent_stats[agent_name]["total_duration"] += metrics.average_duration_ms * metrics.executions
        
        # Create performance metrics
        performance_data = []
        for agent_name, stats in agent_stats.items():
            if stats["executions"] > 0:
                performance_data.append({
                    "Agent": agent_name.replace("_", " ").title(),
                    "Executions": stats["executions"],
                    "Success Rate": f"{stats['successes'] / stats['executions'] * 100:.1f}%",
                    "Avg Duration (ms)": f"{stats['total_duration'] / stats['executions']:.0f}",
                    "Total Failures": stats["failures"]
                })
        
        if performance_data:
            df_performance = pd.DataFrame(performance_data)
            st.dataframe(df_performance, use_container_width=True)
            
            # Performance charts
            col1, col2 = st.columns(2)
            
            with col1:
                # Success rate chart
                fig_success = px.bar(
                    df_performance,
                    x="Agent",
                    y="Success Rate",
                    title="Agent Success Rates"
                )
                st.plotly_chart(fig_success, width='stretch')
            
            with col2:
                # Duration chart
                df_performance["Duration"] = df_performance["Avg Duration (ms)"].str.replace(" ms", "").astype(float)
                fig_duration = px.bar(
                    df_performance,
                    x="Agent",
                    y="Duration",
                    title="Average Processing Duration (ms)"
                )
                st.plotly_chart(fig_duration, width='stretch')
    
    def render_escalations_tab(self):
        """Render escalations tab"""
        results = st.session_state.processing_results
        
        escalated_results = [r for r in results if r.overall_status == ProcessingStatus.ESCALATED]
        
        if not escalated_results:
            st.success("🎉 No escalations! All invoices processed successfully.")
            return
        
        st.warning(f"⚠️ {len(escalated_results)} invoices require attention")
        
        for result in escalated_results:
            with st.expander(f"🚨 {result.file_name} - {result.escalation_reason}"):
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write("**Invoice Details:**")
                    if result.invoice_data:
                        st.write(f"- Customer: {result.invoice_data.customer_name}")
                        st.write(f"- Amount: ${result.invoice_data.total:.2f}")
                        st.write(f"- Invoice #: {result.invoice_data.invoice_number}")
                
                with col2:
                    st.write("**Escalation Details:**")
                    st.write(f"- Reason: {result.escalation_reason}")
                    st.write(f"- Human Review: {'Yes' if result.human_review_required else 'No'}")
                    if result.human_review_notes:
                        st.write(f"- Notes: {result.human_review_notes}")
    
    def render_analytics_tab(self):
        """Render analytics tab"""
        results = st.session_state.processing_results
        
        if not results:
            st.info("No processing results available.")
            return
        
        # Risk distribution
        risk_levels = []
        amounts = []
        customers = []
        
        for result in results:
            if result.risk_assessment and result.invoice_data:
                risk_levels.append(result.risk_assessment.risk_level.value)
                amounts.append(result.invoice_data.total)
                customers.append(result.invoice_data.customer_name)
        
        if risk_levels:
            # Risk vs Amount scatter plot
            fig_risk = px.scatter(
                x=amounts,
                y=risk_levels,
                hover_data=[customers],
                title="Risk Level vs Invoice Amount",
                labels={"x": "Invoice Amount ($)", "y": "Risk Level"}
            )
            st.plotly_chart(fig_risk, width='stretch')
        
        # Processing efficiency metrics
        processing_times = []
        statuses = []
        
        for result in results:
            duration = (result.updated_at - result.created_at).total_seconds() / 60
            processing_times.append(duration)
            statuses.append(result.overall_status.value)
        
        if processing_times:
            fig_efficiency = px.histogram(
                x=processing_times,
                color=statuses,
                title="Processing Time Distribution",
                labels={"x": "Processing Time (minutes)", "y": "Count"}
            )
            st.plotly_chart(fig_efficiency, width='stretch')
    
    def show_workflow_diagram(self):
        """Show workflow diagram"""
        st.subheader("🔄 AI Agent Workflow")
        
        # Create a simple workflow visualization
        workflow_steps = [
            "📄 Document Agent → Extract invoice data using AI",
            "✅ Validation Agent → Validate against purchase orders",
            "🛡️ Risk Agent → Assess fraud risk and compliance",
            "💳 Payment Agent → Make payment decisions",
            "📋 Audit Agent → Generate compliance records",
            "⚠️ Escalation Agent → Handle exceptions"
        ]
        
        for i, step in enumerate(workflow_steps, 1):
            st.write(f"{i}. {step}")
        
        st.info("💡 The workflow uses intelligent routing based on validation results, risk scores, and business rules.")
    
    def show_health_check(self):
        """Show system health check"""
        if not self.workflow:
            st.error("Workflow not initialized")
            return
        
        with st.spinner("Performing health check..."):
            try:
                # This would be async in a real implementation
                health_status = asyncio.run(self.workflow.health_check())
                
                if health_status["workflow_status"] == "healthy":
                    st.success("✅ All systems operational")
                else:
                    st.warning(f"⚠️ System status: {health_status['workflow_status']}")
                
                # Show agent status
                st.subheader("Agent Status")
                for agent_name, status in health_status["agents"].items():
                    if status.get("status") == "healthy":
                        st.success(f"✅ {agent_name.replace('_', ' ').title()}")
                    else:
                        st.error(f"❌ {agent_name.replace('_', ' ').title()}: {status.get('status', 'Unknown')}")
                
            except Exception as e:
                st.error(f"❌ Health check failed: {str(e)}")
    
    def run(self):
        """Run the main application"""
        self.render_header()
        
        # Initialize workflow
        if not self.initialize_workflow():
            st.stop()
        
        # Render sidebar and get controls
        controls = self.render_sidebar()
        
        # Handle processing
        if controls["process_button"] and controls["selected_files"]:
            asyncio.run(self.process_invoices_async(
                controls["selected_files"],
                controls["workflow_type"],
                controls["priority_level"],
                controls["max_concurrent"]
            ))
            st.rerun()
        
        # Render main dashboard
        self.render_main_dashboard()


# Run the application
if __name__ == "__main__":
    app = InvoiceProcessingApp()
    app.run()