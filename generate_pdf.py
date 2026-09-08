import os
import sys
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, HRFlowable, Table, TableStyle, KeepTogether

def create_pdf(filename="Chat_Questions_And_Answers.pdf"):
    doc = SimpleDocTemplate(
        filename,
        pagesize=letter,
        rightMargin=40,
        leftMargin=40,
        topMargin=40,
        bottomMargin=40
    )
    
    styles = getSampleStyleSheet()
    
    # Custom Styles
    primary_color = colors.HexColor("#1E3A8A")   # Deep Blue
    secondary_color = colors.HexColor("#2563EB") # Bright Blue
    dark_gray = colors.HexColor("#1F2937")       # Dark Body Text
    light_bg = colors.HexColor("#F8FAFC")        # Soft Gray Background
    border_color = colors.HexColor("#E2E8F0")    # Border Accent

    title_style = ParagraphStyle(
        'DocTitle',
        parent=styles['Heading1'],
        fontName='Helvetica-Bold',
        fontSize=20,
        leading=24,
        textColor=primary_color,
        spaceAfter=6
    )
    
    subtitle_style = ParagraphStyle(
        'DocSubtitle',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=10,
        leading=14,
        textColor=colors.HexColor("#4B5563"),
        spaceAfter=15
    )

    section_heading = ParagraphStyle(
        'SectionHeading',
        parent=styles['Heading2'],
        fontName='Helvetica-Bold',
        fontSize=13,
        leading=17,
        textColor=secondary_color,
        spaceBefore=14,
        spaceAfter=6
    )

    q_style = ParagraphStyle(
        'QuestionStyle',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=10,
        leading=14,
        textColor=colors.HexColor("#111827"),
        spaceAfter=4
    )

    a_style = ParagraphStyle(
        'AnswerStyle',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=9.5,
        leading=14,
        textColor=dark_gray,
        spaceAfter=6
    )

    box_text = ParagraphStyle(
        'BoxText',
        parent=styles['Normal'],
        fontName='Helvetica-Oblique',
        fontSize=9.5,
        leading=14,
        textColor=colors.HexColor("#1E293B")
    )

    story = []

    # Title Block
    story.append(Paragraph("Application Q&A & Technical Project Summary", title_style))
    story.append(Paragraph("<b>Project:</b> Adaptive Self-Verifying Lung Cancer AI &nbsp;|&nbsp; <b>Live App:</b> <a href='https://112005.streamlit.app/'>https://112005.streamlit.app/</a>", subtitle_style))
    story.append(HRFlowable(width="100%", thickness=1.5, color=secondary_color, spaceAfter=15))

    # Helper function for Q&A Card
    def make_card(q_text, a_content, is_box=False):
        card_story = []
        card_story.append(Paragraph(f"<b>Q: {q_text}</b>", q_style))
        card_story.append(Spacer(1, 3))
        
        if isinstance(a_content, str):
            p_style = box_text if is_box else a_style
            card_story.append(Paragraph(a_content, p_style))
        elif isinstance(a_content, list):
            for item in a_content:
                card_story.append(Paragraph(item, a_style))
                card_story.append(Spacer(1, 2))
                
        t = Table([[card_story]], colWidths=[532])
        t.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, -1), light_bg),
            ('BOX', (0, 0), (-1, -1), 0.75, border_color),
            ('PADDING', (0, 0), (-1, -1), 10),
            ('VALIGN', (0, 0), (-1, -1), 'TOP'),
        ]))
        return t

    # SECTION 1
    story.append(Paragraph("1. Vercel Deployment Troubleshooting & Streamlit Hosting", section_heading))
    q1 = "I'm deploying this project on Vercel, but it shows an error: <i>'Error: Found app.py but it does not export a top-level app or application'</i>."
    a1 = [
        "<b>Root Cause:</b> Vercel expects Python files in the root (like <code>app.py</code>) to be stateless WSGI/ASGI serverless endpoints exporting an <code>app</code> object (e.g. Flask or FastAPI).",
        "Streamlit applications (<code>streamlit run app.py</code>) require a stateful background Python server process and WebSockets, which Vercel Serverless Functions do not support. Additionally, heavy ML dependencies like PyTorch (<code>torch</code>) exceed Vercel's 250 MB function limit.",
        "<b>Resolution & Deployment:</b> Successfully deployed the Streamlit app to <b>Streamlit Community Cloud</b>.",
        "<b>Live Production URL:</b> <a href='https://112005.streamlit.app/'>https://112005.streamlit.app/</a> (Verified active with 3D CT slice viewers, Grad-CAM XAI, Self-Verification engine, and MC uncertainty calibration)."
    ]
    story.append(make_card(q1, a1))
    story.append(Spacer(1, 10))

    # SECTION 2
    story.append(Paragraph("2. Project Highlight (2-Minute Link Walkthrough)", section_heading))
    q2 = "If we only have two minutes with that link, what should we look at, and why? Point us at the specific file, feature, or decision you are proudest of."
    a2 = "Check out the <b>Self-Verification & Clinical Diagnosis tab</b> (<code>src/self_verification.py</code>), which cross-checks predicted malignancy risk against radiological nodule features and Monte Carlo uncertainty to automatically flag potential false positives. It turns a black-box 3D neural network into an adaptive, self-auditing decision support system built for high-stakes clinical safety."
    story.append(make_card(q2, a2, is_box=True))
    story.append(Spacer(1, 10))

    # SECTION 3
    story.append(Paragraph("3. Confidently Wrong LLM Output & Code Guardrails", section_heading))
    q3 = "Describe a time an LLM gave you a confidently wrong answer inside something you were building. How did you catch it, and what did you change so it would not happen again?"
    a3 = [
        "<b>What it got wrong:</b> While writing code to process 3D medical CT volumes using <code>pydicom</code> and <code>SimpleITK</code>, an LLM generated plausible-looking code that assumed array dimensions were structured as <code>(C, H, W, D)</code> instead of standard spatial <code>(D, H, W)</code>. It also hallucinated a non-existent parameter name for Hounsfield Unit windowing.",
        "<b>How I caught it:</b> My unit tests failed immediately during tensor shape assertions (<code>pytest</code>), and visual slice overlays showed rotated, corrupted organ geometry during manual inspection.",
        "<b>What I put in place afterwards:</b><br/>"
        "1. Created strict automated unit tests and <code>pytest</code> shape/range assertions for all data ingestion pipelines.<br/>"
        "2. Added Pydantic schema validation on data structures so unexpected shapes or hallucinated API parameters throw runtime errors immediately instead of failing silently downstream."
    ]
    story.append(make_card(q3, a3))
    story.append(Spacer(1, 10))

    # SECTION 4
    story.append(Paragraph("4. 300-Page Annual Financial Report Audit (1-Hour Challenge)", section_heading))
    q4 = "You are handed a 300-page annual report and an AI system that has extracted the financials from it. You have one hour to decide whether a banker can put their name on that output. What do you check, and in what order?"
    a4 = "First, I’d run automated math checks on the extracted data (<code>Assets = Liabilities + Equity</code>, subtotals matching line items). If basic arithmetic fails, the AI pipeline is unreliable and I can stop right there without wasting the remaining 55 minutes.<br/><br/>If the math holds, I’d spot-check the 10 biggest numbers (Revenue, Net Income, Debt) directly against the raw PDF pages, double-checking units (millions vs. thousands) and multi-column alignment where LLMs often trip up. Finally, I'd check low-confidence flags and footnote restatements before signing off."
    story.append(make_card(q4, a4))
    story.append(Spacer(1, 10))

    # SECTION 5
    story.append(Paragraph("5. Self-Taught Endeavor in the Last 6 Months (Ungraded)", section_heading))
    q5 = "What is something you taught yourself in the last six months that nobody assigned to you? Does not have to be technical."
    a5 = [
        "<b>Option A — Non-Technical (Fitness & Posture Mobility):</b><br/>"
        "• <i>What it was:</i> Progressive calisthenics and shoulder/spine mobility training.<br/>"
        "• <i>What set me off:</i> I noticed how stiff my posture and back were getting after sitting at a desk for long coding sessions, and I wanted to improve physical health without needing a gym membership.<br/>"
        "• <i>How far I got:</i> Built a daily 20-minute mobility routine, fixed posture, and worked up from basic bodyweight exercises to holding a controlled 10-second freestanding handstand.<br/>",
        "<b>Option B — Technical (AI Calibration & Uncertainty Quantification):</b><br/>"
        "• <i>What it was:</i> Post-hoc confidence calibration (Temperature Scaling) and Monte Carlo Dropout to measure epistemic and aleatoric uncertainty.<br/>"
        "• <i>What set me off:</i> Noticed how black-box neural networks often output 99% confidence even when predicting on noisy or ambiguous data.<br/>"
        "• <i>How far I got:</i> Implemented temperature scaling from scratch and built a pipeline that separates model uncertainty from data noise into a live clinical UI."
    ]
    story.append(make_card(q5, a5))
    story.append(Spacer(1, 10))

    # SECTION 6
    story.append(Paragraph("6. Strong Opinion / Open Closing Thoughts", section_heading))
    q6 = "Anything we did not ask that we should know? A constraint, an unusual background, a question for us, a strong opinion about something."
    a6 = "<b>Strong opinion on AI engineering:</b> High accuracy is only half the battle; the real engineering challenge is building safety systems <i>around</i> the model. Most real-world AI failures aren't caused by low accuracy, but by models being confidently wrong without any uncertainty bounds or verification rails."
    story.append(make_card(q6, a6, is_box=True))

    doc.build(story)
    print(f"PDF successfully created at: {os.path.abspath(filename)}")

if __name__ == "__main__":
    create_pdf()
