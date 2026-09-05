```{=html}
<div class="post-list list">
<% for (const item of items) { %>
  <article class="post-card" <%= metadataAttrs(item) %>>
    <time class="post-card__date listing-date"><%= item.date %></time>

    <h2 class="post-card__title listing-title">
      <a href="<%- item.path %>"><%- item.title %></a>
    </h2>

    <% if (item.image) { %>
    <a class="post-card__image listing-image"
       href="<%- item.path %>"
       aria-label="Read <%- item.title %>">
      <img src="<%- item.image %>"
           alt="<%- item['image-alt'] || '' %>"
           loading="lazy">
    </a>
    <% } %>

    <div class="post-card__description listing-description">
      <%= item.description %>
    </div>
  </article>
<% } %>
</div>
```
