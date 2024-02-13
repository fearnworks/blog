---
layout: default
title: Welcome to the Fearnworks Blog!
---

# Welcome to the Fearnworks Blog!

## Heavy Interest in Data Engineering, Analytics, AI/ML, Systems Engineering, and Knowledge Management.

This is the place where I share my thoughts, findings, and expertise on various topics related to data engineering, analytics, artificial intelligence, machine learning, systems engineering, and knowledge management.

### Latest Posts

<ul>
  {% for post in site.posts limit:5 %}
    <li>
      <h2><a href="{{ post.url | prepend: site.baseurl }}">{{ post.title }}</a></h2>
      <p>{{ post.excerpt }}</p>
      <p><a href="{{ post.url | prepend: site.baseurl }}">Read more...</a></p>
    </li>
  {% endfor %}
</ul>

<!-- ### About Me

[Learn more about me and my work.](/about/) -->

### Contact

[Get in touch with me for any inquiries or collaborations.](https://twitter.com/fearnworks)


