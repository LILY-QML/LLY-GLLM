import pandas as pd
import numpy as np
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
from qiskit.visualization import plot_histogram
import os
import seaborn as sns

class Visual:
    def __init__(self, final_summary, comparison_df, circuits, num_iterations, qubits, depth):
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

        # Add content
        self.add_title_page(story)
        self.add_comparison_section(story)

        # Add probability distributions
        self.add_probability_distributions(story)

        # Build the PDF
        doc.build(story)

    def add_title_page(self, story):
        title = "Quantum Circuit Report"
        story.append(Paragraph(title, self.styles['Title']))
        story.append(Spacer(1, 12))

    def add_comparison_section(self, story):
        story.append(Paragraph("Comparison of Initial and Final States", self.styles['Heading2']))
        story.append(Spacer(1, 12))
        
        # Example: Adding a table with formatted data
        comparison_data = [[str(i) for i in row] for row in self.comparison_df.values.tolist()]
        table = Table([self.comparison_df.columns.tolist()] + comparison_data)
        table.setStyle(
            TableStyle(
                [
                    ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
                    ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                    ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                    ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                    ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                    ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
                ]
            )
        )
        story.append(table)
        story.append(Spacer(1, 20))

    def add_probability_distributions(self, story):
        story.append(Paragraph("Probability Distributions", self.styles['Heading2']))
        story.append(Spacer(1, 12))
        
        for word, data in self.final_summary.items():
            # Assuming data is a dictionary with probability distributions
            probabilities = data.get('probabilities', {})
            plt.figure(figsize=(10, 6))
            plt.bar(probabilities.keys(), probabilities.values(), color='blue')
            plt.xlabel('States')
            plt.ylabel('Probability')
            plt.title(f'Probability Distribution for {word}')
            plt.xticks(rotation=45)
            plt.tight_layout()
            image_path = f"{word}_distribution.png"
            plt.savefig(image_path)
            plt.close()

            # Add image to PDF
            story.append(Paragraph(f"Probability Distribution for {word}", self.styles['Heading3']))
            story.append(Image(image_path, width=400, height=300))
            story.append(Spacer(1, 20))

class TitlePage:
    def __init__(
        self, title, subtitle, copyright_info, description, date, additional_info
    ):
        self.title = title
        self.subtitle = subtitle
        self.copyright_info = copyright_info
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
        copyright_paragraph = Paragraph(self.copyright_info, normal_style)
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
                copyright_paragraph,
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


class TableOfContents:
    def __init__(self, contents):
        self.contents = contents

    def build(self, story, styles):
        toc_title_style = styles["Heading2"]
        toc_title = Paragraph("Table of Contents", toc_title_style)

        # Create paragraphs for each item in the table of contents
        toc_entries = [toc_title]
        toc_style = ParagraphStyle(
            "toc",
            parent=styles["Normal"],
            spaceBefore=5,
            spaceAfter=5,
            leftIndent=20,
            fontSize=12,
        )

        for entry in self.contents:
            toc_entry = Paragraph(entry, toc_style)
            toc_entries.append(toc_entry)

        # Adding content to the story
        story.extend(toc_entries + [Spacer(1, 40)])


class OptimizationMethod:
    def __init__(self, title, description, use_case):
        self.title = title
        self.description = description
        self.use_case = use_case

    def build(self, story, styles):
        method_title = Paragraph(f"{self.title}", styles["Heading3"])
        method_description = Paragraph(
            f"<b>Description:</b> {self.description}", styles["Normal"]
        )
        method_use_case = Paragraph(
            f"<b>Use Case:</b> {self.use_case}", styles["Normal"]
        )

        # Adding content to the story
        story.extend(
            [
                method_title,
                Spacer(1, 10),
                method_description,
                Spacer(1, 5),
                method_use_case,
                Spacer(1, 20),
            ]
        )


class ComparisonSection:
    def __init__(self, content):
        self.content = content

    def build(self, story, styles):
        comparison_title_style = styles["Heading2"]
        comparison_title = Paragraph("Comparison Between Methods", comparison_title_style)

        comparison_content = Paragraph(self.content, styles["Normal"])

        # Adding content to the story
        story.extend(
            [
                comparison_title,
                Spacer(1, 20),
                comparison_content,
                Spacer(1, 40),
            ]
        )


class FinalResultsSection:
    def __init__(self, content):
        self.content = content

    def build(self, story, styles):
        final_results_title_style = styles["Heading2"]
        final_results_title = Paragraph("Final Results", final_results_title_style)

        final_results_content = Paragraph(self.content, styles["Normal"])

        # Adding content to the story
        story.extend(
            [
                final_results_title,
                Spacer(1, 20),
                final_results_content,
                Spacer(1, 40),
            ]
        )
