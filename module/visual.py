import pandas as pd
import matplotlib.pyplot as plt
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    PageBreak,
    Image,
    Table,
    TableStyle,
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from datetime import datetime
import os


class Visual:
    def __init__(
        self,
        final_summary,
        comparison_df,
        circuits,
        num_iterations,
        qubits,
        depth,
    ):
        self.final_summary = final_summary
        self.comparison_df = comparison_df
        self.circuits = circuits
        self.num_iterations = num_iterations
        self.qubits = qubits
        self.depth = depth
        self.styles = getSampleStyleSheet()

    def generate_report(self, filename="QuantumCircuitReport.pdf"):
        # Create the document
        doc = SimpleDocTemplate(filename, pagesize=letter)
        story = []

        # Title Page
        self.add_title_page(story)

        # Initial and Final States Comparison Table
        self.add_comparison_table(story)

        # Probability Distributions and Loss Functions
        self.add_probability_distributions(story)

        # Build the PDF
        doc.build(story)

    def add_title_page(self, story):
        # Create the title page
        title_page = TitlePage(
            title="LLY-DML Quantum Circuit Report",
            subtitle="Part of the LILY Project",
            description="""<hr/>
            This report showcases the quantum circuit training results of the LLY-DML model.<br/>
            The results include a comparison of initial and final states, along with probability distributions and loss functions.<br/>
            <hr/>""",
            date=datetime.now().strftime("%d.%m.%Y"),
            additional_info="""<b>Version:</b> 1.0<br/>
            <b>Contact:</b> info@lilyqml.de<br/>
            <b>Website:</b> <a href="http://lilyqml.de">lilyqml.de</a><br/>""",
        )
        title_page.build(story, self.styles)

    def add_comparison_table(self, story):
        # Section: Comparison Between Initial and Final States
        story.append(
            Paragraph(
                "<a name='section3'/>Comparison Between Initial and Final States",
                self.styles["Heading2"],
            )
        )
        story.append(Spacer(1, 20))

        # Add the comparison table
        table_data = [
            [str(i) for i in row] for row in self.comparison_df.round(4).values.tolist()
        ]  # Convert all elements to strings
        comparison_table = Table([self.comparison_df.columns.tolist()] + table_data)
        comparison_table.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, 0), colors.grey),
                    ("TEXTCOLOR", (0, 0), (-1, 0), colors.whitesmoke),
                    ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                    ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                    ("BOTTOMPADDING", (0, 0), (-1, 0), 12),
                    ("BACKGROUND", (0, 1), (-1, -1), colors.beige),
                ]
            )
        )
        story.append(comparison_table)
        story.append(Spacer(1, 20))
        story.append(PageBreak())

    def add_probability_distributions(self, story):
        # Section: Probability Distributions and Loss Functions
        story.append(
            Paragraph(
                "<a name='section4'/>Probability Distributions and Loss Functions",
                self.styles["Heading2"],
            )
        )
        story.append(Spacer(1, 20))

        for summary in self.final_summary:
            word = summary["Wort"]
            counts = summary["Counts"]
            loss = summary["Loss"]

            # Plot Probability Distribution
            plt.figure(figsize=(10, 5))
            plt.bar(counts.keys(), counts.values())
            plt.xlabel('State')
            plt.ylabel('Probability')
            plt.title(f'Probability Distribution for {word}')
            plt.xticks(rotation=90)
            plt.tight_layout()

            # Save and append the image to the PDF
            prob_dist_path = f"var/{word}_prob_dist.png"
            plt.savefig(prob_dist_path)
            plt.close()
            story.append(Image(prob_dist_path, width=400, height=200))
            story.append(Spacer(1, 20))

            # Plot Loss Function
            plt.figure(figsize=(10, 5))
            plt.plot(loss)
            plt.xlabel('Iteration')
            plt.ylabel('Loss')
            plt.title(f'Loss Function for {word}')
            plt.tight_layout()

            # Save and append the image to the PDF
            loss_func_path = f"var/{word}_loss_func.png"
            plt.savefig(loss_func_path)
            plt.close()
            story.append(Image(loss_func_path, width=400, height=200))
            story.append(Spacer(1, 20))

        story.append(PageBreak())


class TitlePage:
    def __init__(self, title, subtitle, description, date, additional_info):
        self.title = title
        self.subtitle = subtitle
        self.description = description
        self.date = date
        self.additional_info = additional_info

    def build(self, story, styles):
        # Custom styles for specific formatting
        title_style = ParagraphStyle(
            "title",
            parent=styles["Title"],
            fontSize=36,
            textColor=colors.HexColor("#000080"),  # Dunkelblau
            spaceAfter=24,  # Abstand nach der Hauptüberschrift
        )
        subtitle_style = ParagraphStyle(
            "subtitle",
            parent=styles["Heading2"],
            fontSize=16,
            spaceAfter=5,  # Weniger Abstand zwischen Untertitel und Copyright-Text
        )
        normal_style = ParagraphStyle(
            "normal",
            parent=styles["Normal"],
            alignment=4,  # Blocksatz
            spaceBefore=10,  # Abstand vor dem Text
        )
        big_heading_style = ParagraphStyle(
            "bigHeading",
            parent=styles["Heading1"],
            fontSize=24,
            textColor=colors.HexColor("#000080"),  # Dunkelblau
            spaceAfter=20,  # Abstand nach der Überschrift
        )

        # Paragraphs with the new styles
        title_paragraph = Paragraph(self.title, title_style)
        subtitle_paragraph = Paragraph(self.subtitle, subtitle_style)
        description_paragraph = Paragraph(self.description, normal_style)
        big_heading_paragraph = Paragraph(
            "QUANTUM LLY-DML TRAINING REPORT", big_heading_style
        )
        additional_info_paragraph = Paragraph(self.additional_info, normal_style)

        date_paragraph = Paragraph(
            f"""<hr/>
            This report shows all data related to the training conducted on: <b>{self.date}</b>
            <hr/>""",
            normal_style,
        )

        # Adding content to the story
        story.extend(
            [
                title_paragraph,
                subtitle_paragraph,
                Spacer(1, 40),  # Adjusted space between copyright and description
                description_paragraph,
                Spacer(1, 20),
                big_heading_paragraph,
                additional_info_paragraph,
                Spacer(1, 40),
                date_paragraph,
                Spacer(1, 40),
                PageBreak(),  # Page break for the table of contents
            ]
        )
